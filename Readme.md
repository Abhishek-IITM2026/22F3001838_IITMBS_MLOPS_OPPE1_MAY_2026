# 22F3001838_IITMBS_MLOPS_OPPE1_MAY_2026

## Overview

This repository contains the implementation of an end-to-end MLOps pipeline developed as part of the IIT Madras BS in Data Science MLOps OPPE.

The project demonstrates:

* Data versioning using DVC
* Google Cloud Storage (GCS) as DVC remote
* Feature management using Feast
* Model training and experiment tracking using MLflow
* Automated CI using GitHub Actions
* Model evaluation and prediction

---

# Project Structure

```
22F3001838_IITMBS_MLOPS_OPPE1_MAY_2026/
│
├── data/
│   ├── raw/
│   │   ├── v0/
│   │   └── v1/
│   └── processed/
│
├── models/
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── predict.py
│
├── feature_repo/
│
├── reports/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── requirements.txt
├── dvc.yaml
└── README.md
```

---

# Prerequisites

* Python 3.11
* Git
* DVC
* Google Cloud SDK
* Google Cloud Storage Bucket
* Feast
* MLflow

---

# Clone Repository

```bash
git clone <repository_url>
cd 22F3001838_IITMBS_MLOPS_OPPE1_MAY_2026
```

---

# Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Initialize Git

```bash
git init
git add .
git commit -m "Initial Commit"
```

---

# Initialize DVC

```bash
dvc init
git add .
git commit -m "Initialize DVC"
```

---

# Configure Google Cloud Storage Remote

```bash
dvc remote add -d storage gs://22f3001838_mlops_oppe1
```

Verify:

```bash
dvc remote list
```

Expected output:

```
storage gs://22f3001838_mlops_oppe1 (default)
```

---

# Add Dataset to DVC

Copy the dataset into:

```
data/raw/v0/
```

Track the dataset:

```bash
dvc add data/raw/v0
```

Commit metadata:

```bash
git add data/raw/v0.dvc .gitignore
git commit -m "Track dataset using DVC"
```

Upload data to GCS:

```bash
dvc push
```

Download data:

```bash
dvc pull
```

---

# Prepare Dataset

```bash
python src/prepare_data.py
```

Output:

```
data/processed/
```

---

# Create Parquet File

If your dataset is in CSV format, convert it to Parquet using Pandas.

```python
import pandas as pd

df = pd.read_csv("data/raw/v0/iris.csv")

df.to_parquet(
    "feature_repo/data/iris.parquet",
    index=False
)
```

Verify:

```python
import pandas as pd

df = pd.read_parquet("feature_repo/data/iris.parquet")

print(df.head())
```

---

# Feast

Initialize Feast repository (if required):

```bash
cd feature_repo
feast init .
```

Apply Feature Store:

```bash
feast apply
```

Materialize Features:

```bash
feast materialize-incremental now
```

List Feature Views:

```bash
feast feature-views list
```

---

# Train Model

```bash
python src/train.py
```

The trained model will be saved inside:

```
models/
```

---

# Hyperparameter Tuning

```bash
python src/tune.py
```

---

# Evaluate Model

```bash
python src/evaluate.py
```

---

# Prediction

```bash
python src/predict.py
```

---

# MLflow

Run training with MLflow logging:

```bash
python mlflow_train.py
```

Start MLflow UI:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

# GitHub Actions

Push repository:

```bash
git add .
git commit -m "Update project"
git push origin main
```

GitHub Actions automatically:

* Installs dependencies
* Pulls data using DVC
* Runs tests
* Trains the model
* Logs experiments

---

# Running Tests

```bash
pytest
```

or

```bash
pytest -v
```

---

# Requirements

```
pandas==2.2.3
numpy==2.2.6
scikit-learn==1.6.1
joblib==1.5.1
dvc==3.63.0
dvc-gs==3.0.2
mlflow==2.22.2
feast==0.65.0
gcsfs==2025.5.1
pyarrow==19.0.1
matplotlib==3.10.3
pytest==8.4.1
cml
```

---

# Useful DVC Commands

Initialize:

```bash
dvc init
```

Track data:

```bash
dvc add data/raw/v0
```

Push data:

```bash
dvc push
```

Pull data:

```bash
dvc pull
```

Status:

```bash
dvc status
```

List remotes:

```bash
dvc remote list
```

---

# Useful Git Commands

```bash
git status
git add .
git commit -m "Commit Message"
git push origin main
git pull origin main
```

---

# Author

**Abhishek Dutta**

IIT Madras BS in Data Science and Applications

MLOps OPPE Project
