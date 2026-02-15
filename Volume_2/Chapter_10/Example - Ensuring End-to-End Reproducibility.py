# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 04:13:06 2025

@author: Admin
"""

import sys
import platform
import random

import joblib
import sklearn
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1) Fix all random seeds (NumPy, Python, and random_state)
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# 2) Load data and prepare features/target
#    (here: Breast Cancer dataset as a pandas DataFrame)
# ============================================================
data = load_breast_cancer(as_frame=True)
X = data.data         # Features as DataFrame
y = data.target       # Target as Series

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=RANDOM_SEED,
    stratify=y
)

# ============================================================
# 3) Store preprocessing + model inside a Pipeline
#    (StandardScaler + LogisticRegression)
# ============================================================
clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000
        )),
    ]
)

# Train the full pipeline
clf.fit(X_train, y_train)

print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# ============================================================
# 4) Log versions of all key dependencies + environment
# ============================================================
print("\n=== Environment and library versions ===")
print("Python version:", sys.version)
print("Platform:", platform.platform())
print("scikit-learn version:", sklearn.__version__)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)

# ============================================================
# 5) Save both model and preprocessor using joblib
#    (the entire Pipeline is serialized)
# ============================================================
joblib.dump(clf, "breast_cancer_pipeline.joblib")

# ============================================================
# 6) Reproducible loading in the SAME environment
#    (training and inference use identical library versions)
# ============================================================
loaded_model = joblib.load("breast_cancer_pipeline.joblib")
print("\nReloaded model test accuracy:",
      loaded_model.score(X_test, y_test))