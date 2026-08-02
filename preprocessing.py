import os
import glob
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_stock(file_path):
    print(f"Processing {os.path.basename(file_path)}")

    df = pd.read_csv(file_path)

    # stock name
    stock_name = os.path.basename(file_path).split("__")[0]
    df["stock_name"] = stock_name

    # timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # sort
    df = df.sort_values("timestamp")

    # forward fill
    df = df.ffill()

    # rolling calculations
    df.set_index("timestamp", inplace=True)

    df["rolling_avg_10"] = (
        df["close"]
        .rolling(window="10min", min_periods=1)
        .mean()
    )

    df["volume_sum_10"] = (
        df["volume"]
        .rolling(window="10min", min_periods=1)
        .sum()
    )

    # target
    df["future_close"] = df["close"].shift(-5)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df = df.dropna()

    df.drop(columns=["future_close"], inplace=True)

    df.reset_index(inplace=True)

    return df


#############################################
# ITERATION 1 (v0)
#############################################

print("\nProcessing v0...\n")

v0_files = glob.glob(os.path.join(RAW_DIR, "v0", "*.csv"))

v0_data = []

for file in sorted(v0_files):
    stock_df = process_stock(file)
    v0_data.append(stock_df)

v0_df = pd.concat(v0_data, ignore_index=True)

print("Total rows in v0:", len(v0_df))

train_v0 = []
test_v0 = []

for stock in v0_df["stock_name"].unique():

    temp = v0_df[v0_df["stock_name"] == stock]

    train_v0.append(temp.iloc[:-20])

    test_v0.append(temp.iloc[-20:])

train_v0 = pd.concat(train_v0, ignore_index=True)
test_v0 = pd.concat(test_v0, ignore_index=True)

train_v0.to_csv(
    os.path.join(PROCESSED_DIR, "train_v0.csv"),
    index=False
)

test_v0.to_csv(
    os.path.join(PROCESSED_DIR, "test_v0.csv"),
    index=False
)

print("Saved train_v0.csv")
print("Saved test_v0.csv")


#############################################
# ITERATION 2 (v0 + v1)
#############################################

print("\nProcessing v1...\n")

v1_files = glob.glob(os.path.join(RAW_DIR, "v1", "*.csv"))

all_data = []

for file in sorted(v0_files + v1_files):
    stock_df = process_stock(file)
    all_data.append(stock_df)

all_df = pd.concat(all_data, ignore_index=True)

print("Total rows in v0+v1:", len(all_df))

train_all = []
test_all = []

for stock in all_df["stock_name"].unique():

    temp = all_df[all_df["stock_name"] == stock]

    train_all.append(temp.iloc[:-20])

    test_all.append(temp.iloc[-20:])

train_all = pd.concat(train_all, ignore_index=True)
test_all = pd.concat(test_all, ignore_index=True)

train_all.to_csv(
    os.path.join(PROCESSED_DIR, "train_all.csv"),
    index=False
)

test_all.to_csv(
    os.path.join(PROCESSED_DIR, "test_all.csv"),
    index=False
)

print("Saved train_all.csv")
print("Saved test_all.csv")

print("\nCompleted Successfully!")

print("\nShapes")
print("train_v0 :", train_v0.shape)
print("test_v0  :", test_v0.shape)
print("train_all:", train_all.shape)
print("test_all :", test_all.shape)
