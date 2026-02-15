# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 02:15:52 2025

@author: Admin
"""

# ============================================================
# Example: Classification with the Adult Income Dataset (OpenML)
# ============================================================

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# ------------------------------------------------------------
# Step 1 — Load the Adult dataset
# ------------------------------------------------------------
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame.copy()

# Separate features and target
X = df.drop(columns=["class"])
y = df["class"]

# Identify column types
categorical_cols = X.select_dtypes(include=["category", "object"]).columns
numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns

# ------------------------------------------------------------
# Step 2 — Build preprocessing pipeline
# ------------------------------------------------------------
# Numerical: impute missing values with median
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical: impute missing values then One-Hot Encode
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)

# ------------------------------------------------------------
# Step 3 — Create full training pipeline (preprocess + model)
# ------------------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# ------------------------------------------------------------
# Step 4 — Train-test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ------------------------------------------------------------
# Step 5 — Train the model
# ------------------------------------------------------------
model.fit(X_train, y_train)

# ------------------------------------------------------------
# Step 6 — Evaluate performance
# ------------------------------------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
