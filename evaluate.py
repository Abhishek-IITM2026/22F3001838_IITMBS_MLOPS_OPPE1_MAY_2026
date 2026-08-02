import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

model = joblib.load("models/best_model.joblib")

test = pd.read_csv("data/processed/test_all.csv")

X = test[
    [
        "rolling_avg_10",
        "volume_sum_10",
    ]
]

y = test["target"]

pred = model.predict(X)

print("Accuracy :", accuracy_score(y, pred))
print("Precision:", precision_score(y, pred))
print("Recall   :", recall_score(y, pred))
print("F1 Score :", f1_score(y, pred))
