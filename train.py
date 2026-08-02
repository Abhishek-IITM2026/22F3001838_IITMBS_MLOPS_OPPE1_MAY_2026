import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------
# Create output folders
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

FEATURES = [
    "rolling_avg_10",
    "volume_sum_10",
]

TARGET = "target"


def train_and_evaluate(train_file, test_file, model_name):
    print(f"\n==============================")
    print(f"Training: {model_name}")
    print(f"==============================")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    # -----------------------------
    # Use a small subset for faster training
    # -----------------------------
    train_df = train_df.sample(
    n=min(5000, len(train_df)),
    random_state=42
    )   

    test_df = test_df.sample(
    n=min(1000, len(test_df)),
    random_state=42
    )
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    model_path = f"models/{model_name}.joblib"
    joblib.dump(model, model_path)

    return {
        "Iteration": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }


# ======================================
# Iteration 1
# ======================================

results_v0 = train_and_evaluate(
    "data/processed/train_v0.csv",
    "data/processed/test_v0.csv",
    "model_v0",
)

# ======================================
# Iteration 2
# ======================================

results_all = train_and_evaluate(
    "data/processed/train_all.csv",
    "data/processed/test_all.csv",
    "model_all",
)

# ======================================
# Save comparison
# ======================================

results = pd.DataFrame([results_v0, results_all])

results.to_csv(
    "reports/results.csv",
    index=False,
)

print("\n==============================")
print("Training Complete")
print("==============================")

print(results)
