import os
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_experiment("StockPrediction")

os.makedirs("models", exist_ok=True)

train = pd.read_csv("data/processed/train_all.csv")
test = pd.read_csv("data/processed/test_all.csv")

train = train.sample(
    n=min(5000, len(train)),
    random_state=42
)

test = test.sample(
    n=min(1000, len(test)),
    random_state=42
    )

FEATURES = [
    "rolling_avg_10",
    "volume_sum_10"
]

TARGET = "target"

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

best_accuracy = -1
best_model = None
best_run_id = None

for n_estimators in [50, 100, 200]:

    for max_depth in [5, 10, None]:

        with mlflow.start_run():

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_run_id = mlflow.active_run().info.run_id

joblib.dump(best_model, "models/best_model.joblib")

print("Best Accuracy:", best_accuracy)
print("Best Run:", best_run_id)

# ----------------------------
# Register Best Model
# ----------------------------

model_uri = f"runs:/{best_run_id}/model"

registered = mlflow.register_model(
    model_uri=model_uri,
    name="StockPredictionModel"
)

print("Registered Model Version:", registered.version)
