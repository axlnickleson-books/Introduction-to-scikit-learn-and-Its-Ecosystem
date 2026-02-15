# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 01:39:23 2025

@author: Admin
"""

# Reproducible end-to-end example
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

# ----- 1) Toy dataset (deterministic) -----
rng = np.random.RandomState(42)

n = 200
df = pd.DataFrame({
    "age": rng.randint(18, 70, size=n),
    "balance": rng.normal(1000, 300, size=n).round(2),
    "job": rng.choice(["admin", "tech", "blue-collar", "services"], size=n),
    "marital": rng.choice(["single", "married", "divorced"], size=n),
    "education": rng.choice(["primary", "secondary", "tertiary"], size=n),
})

# Binary target with a bit of structure (deterministic)
y = ((df["age"] > 40).astype(int) | (df["balance"] > 1100).astype(int)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.25, random_state=42, stratify=y
)

# ----- 2) Columns -----
numeric = ["age", "balance"]
categorical = ["job", "marital", "education"]

# ----- 3) Preprocessor (scaling + one-hot) -----
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

# ----- 4) Pipeline with LogisticRegression (seeded) -----
model = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42, solver="liblinear")),
])

# ----- 5) Train, evaluate, and persist -----
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

dump(model, "production_model.joblib")
print("Saved to production_model.joblib")

# (Optional) Load and predict to verify same behavior
loaded = load("production_model.joblib")
pred2 = loaded.predict(X_test)
print("Loaded model same preds:", np.array_equal(pred, pred2))
