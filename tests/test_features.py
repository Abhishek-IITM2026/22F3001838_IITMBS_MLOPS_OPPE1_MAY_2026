import pandas as pd

df = pd.read_csv("data/processed/train_all.csv")

def test_rolling_avg():

    assert (df["rolling_avg_10"] > 0).all()

def test_volume_sum():

    assert (df["volume_sum_10"] >= 0).all()
