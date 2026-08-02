import pandas as pd

df = pd.read_csv("data/processed/feast_training.csv")


def test_rolling_avg_exists():
    assert "rolling_avg_10" in df.columns


def test_volume_sum_exists():
    assert "volume_sum_10" in df.columns


def test_rolling_avg_no_nulls():
    assert df["rolling_avg_10"].notna().all()


def test_volume_sum_non_negative():
    assert (df["volume_sum_10"] >= 0).all()
