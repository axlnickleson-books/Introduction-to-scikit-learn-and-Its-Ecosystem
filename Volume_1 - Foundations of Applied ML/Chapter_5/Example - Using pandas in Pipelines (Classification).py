# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 01:45:24 2025

@author: Admin
"""

# Step 1 — Imports
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Step 2 — Load Adult dataset
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame

print("Dataset preview:")
print(df.head())

# Target column is "class" — convert to binary {0,1}
df["class"] = df["class"].astype("category")

X = df.drop("class", axis=1)
y = df["class"]

# Step 3 — Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Step 4 — Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Step 5 — Build Pipeline
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

# Step 6 — Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7 — Fit model
clf.fit(X_train, y_train)

# Step 8 — Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # needed for AUC

# Step 9 — Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average="macro"))
print("Recall (macro):", recall_score(y_test, y_pred, average="macro"))
print("F1-score (macro):", f1_score(y_test, y_pred, average="macro"))

# AUC score (for binary classification)
print("ROC AUC:", roc_auc_score(y_test.cat.codes, y_proba))

# Detailed per-class breakdown
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
