# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:00:07 2025

@author: Admin
"""


import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# For reproducibility
np.random.seed(42)

# ============================================
# 1. Load dataset as a pandas DataFrame
# ============================================
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target

# ============================================
# 2. Create simple categorical features
#    (to demonstrate mixed-type preprocessing)
# ============================================
# Bin "mean radius" into categories
X["radius_group"] = pd.cut(
    X["mean radius"],
    bins=[0, 12, 18, np.inf],
    labels=["small", "medium", "large"]
)

# Bin "mean texture" into high / low
median_texture = X["mean texture"].median()
X["texture_group"] = np.where(
    X["mean texture"] >= median_texture,
    "high_texture",
    "low_texture"
)

# ============================================
# 3. Define numeric and categorical columns
# ============================================
numeric_features = ["mean radius", "mean texture", "mean area"]
categorical_features = ["radius_group", "texture_group"]

# ============================================
# 4. Build the preprocessing transformer
# ============================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ============================================
# 5. Build the full pipeline (preprocess + model)
# ============================================
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# ============================================
# 6. Train/test split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================
# 7. Fit the pipeline
# ============================================
pipeline.fit(X_train, y_train)

# ============================================
# 8. Evaluate the model
# ============================================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Basic metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("=== Evaluation Metrics ===")
print(f"Accuracy:          {acc:.4f}")
print(f"Precision:         {prec:.4f}")
print(f"Recall:            {rec:.4f}")
print(f"F1-score:          {f1:.4f}")
print(f"ROC AUC:           {roc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

