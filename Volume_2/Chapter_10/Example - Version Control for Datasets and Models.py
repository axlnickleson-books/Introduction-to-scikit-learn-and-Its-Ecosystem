# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 04:30:12 2025

@author: Admin
"""

import json
import hashlib
from datetime import datetime

import joblib
import sklearn
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------
# 1. Reproducible setup
# --------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------
# 2. Load dataset
# --------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=RANDOM_SEED,
    stratify=y
)

# --------------------------
# 3. Build and train pipeline
# --------------------------
clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000
        )),
    ]
)

clf.fit(X_train, y_train)

print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# --------------------------
# 4. Compute dataset checksum
# --------------------------
def compute_sha256(df):
    """
    Compute a reproducible SHA-256 hash of a pandas DataFrame.
    This serializes the data to CSV (without index) and hashes the bytes.
    """
    df_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(df_bytes).hexdigest()

dataset_hash = compute_sha256(X)

# --------------------------
# 5. Define dataset + model versions
# --------------------------
DATASET_VERSION = "v2.3"
MODEL_VERSION = "1.0.0"

# --------------------------
# 6. Save the trained pipeline
# --------------------------
model_filename = f"model_{MODEL_VERSION}.joblib"
joblib.dump(clf, model_filename)

# --------------------------
# 7. Collect metadata
# --------------------------
metadata = {
    # Dataset info
    "dataset_version": DATASET_VERSION,
    "dataset_sha256": dataset_hash,

    # Model info
    "model_version": MODEL_VERSION,
    "model_filename": model_filename,
    "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    # Pipeline structure
    "pipeline_steps": [name for name, _ in clf.steps],
    "model_class": clf.named_steps["logreg"].__class__.__name__,
    "model_params": clf.named_steps["logreg"].get_params(),

    # Preprocessing details
    "scaler_class": clf.named_steps["scaler"].__class__.__name__,
    "scaler_params": clf.named_steps["scaler"].get_params(),

    # Features used
    "features_used": list(X.columns),

    # Dependency versions
    "scikit_learn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
}

# --------------------------
# 8. Save metadata as JSON
# --------------------------
metadata_filename = f"metadata_{MODEL_VERSION}.json"
with open(metadata_filename, "w") as f:
    json.dump(metadata, f, indent=4)

print("Model and metadata saved successfully.")
print("Model file:", model_filename)
print("Metadata file:", metadata_filename)