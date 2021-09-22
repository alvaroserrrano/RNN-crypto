# import libraries
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Keras libraries
from tensorflow.keras.models import Sequential

PRECEDING_PRICES = 60  # preceeding 60 days prices
FUTURE_PRICES_PREDICT = 3  # how far into the future to predict
TO_PREDICT = "Bitcoin"


# read datasets and join them
data_directory: Path = Path("./cryptocurrency-historical-prices")
main_df = pd.DataFrame()
for file in data_directory.iterdir():
    if file.is_file() and file.suffix == ".csv":
        cryptocurrency_name = file.name.split("_")[1].split(".")[
            0
        ]  # Bitcoin, Aave, Cardano, Solana, Dogecoin, etc.
    df = pd.read_csv(
        f"{data_directory.name}/{file.name}",
        names=[
            "SNo",
            "Name",
            "Symbol",
            "Date",
            "High",
            "Low",
            "Open",
            "Close",
            "Volume",
            "Marketcap",
        ],
        parse_dates=True,
    )
    # rename to distinguish the cryptocurrency we are working with
    df.rename(
        columns={
            "Close": f"{cryptocurrency_name}_close",
            "Volume": f"{cryptocurrency_name}_volume",
        },
        inplace=True,
    )
    # set time as index so we can join the dataframes
    df.set_index("Date", inplace=True)
    # drop columns we are not interested in
    df = df[[f"{cryptocurrency_name}_close", f"{cryptocurrency_name}_volume"]]
    # df.fillna(method="ffill", inplace=True)
    main_df = df if len(main_df) == 0 else main_df.join(df)
# use previous valid value if there are gaps in the data
main_df.fillna(method="ffill", inplace=True)
# main_df.fillna(method="bfill", inplace=True)
main_df.dropna(how="all", inplace=True)
print(main_df.shape)
print(main_df.head())


def make_order_decision(current_price, future_price):
    if float(current_price) < float(future_price):
        return 1  # buy order
    else:
        return 0


main_df["future_price_to_predict"] = main_df[f"{TO_PREDICT}_close"].shift(
    -FUTURE_PRICES_PREDICT
)  # negative to shift columnn up

# main_df["future_price_to_predict"] = main_df["future_price_to_predict"].astype(float)
main_df["order_decision"] = list(
    map(
        make_order_decision,
        main_df[f"{TO_PREDICT}_close"],
        main_df["future_price_to_predict"],
    )
)
main_df = main_df.dropna()
print(main_df.head())


# normalize data except for the order_decision column
def normalize_and_scale_df(df):
    # no longer need future_price_to_predict column
    df = df.drop("future_price_to_predict", axis=1)

    for column in df.columns:
        if column != "order_decision":
            # TODO why is order_decision being included
            # normalize data based on percentrage
            df[column] = df[column].pct_change()
            df.dropna(inplace=True)  # TODO fix bug here => Empty dataframe
            # print(f"Printing from normalize_and_scale_df {df}")
            # scale values between 0 and 1
            df[column] = preprocessing.scale(df[column].values)
            # min_max_scaler = preprocessing.MinMaxScaler()
            # data_scaled = min_max_scaler.fit_transform(df[column].values.reshape(-1, 1))
            # df[column] = data_scaled

    df.dropna(inplace=True)

    predictions_sequence = []
    previous_days_sequence = deque(maxlen=PRECEDING_PRICES)

    for value in df.values:
        # do not include the target in the sequence
        previous_days_sequence.append([i for i in value[:-1]])
        # only keep last PRECEDING_PRICES observations
        if len(previous_days_sequence) == PRECEDING_PRICES:
            print(value[-1])
            predictions_sequence.append(
                [np.array(previous_days_sequence), value[-1]]
            )
    # shuffle sequential data for good measure
    random.shuffle(predictions_sequence)

    buy_orders = []
    not_buy_orders = []

    for sequence, order_decision in predictions_sequence:
        if order_decision == 1:
            buy_orders.append([sequence, order_decision])
        else:
            not_buy_orders.append([sequence, order_decision])

    random.shuffle(buy_orders)
    random.shuffle(not_buy_orders)

    # ensure both buy and not buy orders are the same length
    shorter_sequence = min(len(buy_orders), len(not_buy_orders))
    buy_orders = buy_orders[:shorter_sequence]
    not_buy_orders = not_buy_orders[:shorter_sequence]

    # combine buy and not buy orders into predictions_sequence
    predictions_sequence = buy_orders + not_buy_orders

    random.shuffle(predictions_sequence)

    x_data = []
    y_data = []

    for sequence, order_decision in predictions_sequence:
        x_data.append(sequence)  # sequence is the input
        y_data.append(order_decision)  # buy or not to buy

    return np.array(x_data), y_data


# data will not be shuffled due to the nature of the data, which is sequential
# taking sequences of data that do not come in the future is likely a bad idea
# make_order_decision will usually be the same for data points 1 minute apart
sorted_dates = sorted(main_df.index.values)
last_5_percent = sorted_dates[-int(len(sorted_dates) * 0.05)]

test_df = main_df[(main_df.index >= last_5_percent)]
print(f"Shape of test data: {test_df.shape}")
main_df = main_df[(main_df.index < last_5_percent)]
print(f"Shape of training data: {main_df.shape}")

# train test split
# normalize and scale training data
x_train, y_train = normalize_and_scale_df(main_df)
# normalize and scale test data
x_test, y_test = normalize_and_scale_df(test_df)

print(f"train data: {len(x_train)} test data: {len(x_test)}")
print(f"Train Dont buys: {y_train.count(0)}, buys: {y_train.count(1)}")
print(f"Test Dont buys: {y_test.count(0)}, buys: {y_test.count(1)}")
