from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo/feature_repo")

entity_df = (
    pd.read_parquet(
        "feature_repo/feature_repo/data/train_small.parquet",
        columns=["stock_name", "timestamp"]
    )
)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
    ],
).to_df()

training_df.to_csv(
    "data/processed/feast_training.csv",
    index=False,
)

print(training_df.head())
print("Rows:", len(training_df))
