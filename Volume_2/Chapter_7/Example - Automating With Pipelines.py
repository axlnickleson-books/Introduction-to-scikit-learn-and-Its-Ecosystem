# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:53:46 2025

@author: Admin
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# ------------------------------------------------------------
# Create a sample dataset with numeric + categorical features
# ------------------------------------------------------------
# Columns:
#   0 = Age (numeric)
#   1 = Fare (numeric)
#   2 = Gender (categorical)
#   3 = Embarked (categorical)
X = np.array([
    [22,   7.25, "male",   "S"],
    [38,  71.28, "female", "C"],
    [26,   7.92, "female", "S"],
    [35,  53.10, "female", "S"],
    [28,   8.05, "male",   "Q"],
    [42,  13.00, "male",   "C"]
], dtype=object)

# Target labels (e.g., 1 = survived, 0 = died)
y = np.array([0, 1, 1, 1, 0, 0])

# ------------------------------------------------------------
# Identify numeric and categorical columns
# ------------------------------------------------------------
numeric_features = [0, 1]
categorical_features = [2, 3]

# ------------------------------------------------------------
# Build numeric and categorical transformers
# ------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # IMPORTANT: handle_unknown="ignore" avoids errors on unseen categories
    ("encoder", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False))
])

# ------------------------------------------------------------
# Combine into a ColumnTransformer
# ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------
# Split into train and test sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Build the full automated pipeline
# ------------------------------------------------------------
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ------------------------------------------------------------
# Train and evaluate
# ------------------------------------------------------------
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))