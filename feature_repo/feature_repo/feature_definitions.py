from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Float32

# Project
project = Project(
    name="feature_repo",
    description="Stock movement prediction feature store",
)

# Entity
stock = Entity(
    name="stock",
    join_keys=["stock_name"],
)

# Offline source
stock_source = FileSource(
    name="stock_source",
    path="data/train_all.parquet",
    timestamp_field="timestamp",
)

# Feature View
stock_features = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=365),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=stock_source,
    online=True,
)
