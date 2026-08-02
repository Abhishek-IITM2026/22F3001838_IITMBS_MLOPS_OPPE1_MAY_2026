import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
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

acc = accuracy_score(y, pred)

cm = confusion_matrix(y, pred)

ConfusionMatrixDisplay(cm).plot()

plt.savefig("confusion.png")

with open("report.md", "w") as f:

    f.write("# Stock Prediction Report\n\n")

    f.write(f"Accuracy : {acc:.4f}\n\n")

    f.write("## Confusion Matrix\n\n")

    f.write("![Confusion Matrix](confusion.png)\n")

