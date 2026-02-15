# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 03:27:18 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ================================
# Step 1: Load dataset
# ================================
data = load_breast_cancer()
X, y = data.data, data.target

# ================================
# Step 2: Split into train/test
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ================================
# Step 3: Build predictor pipeline
# ================================
clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),          # Transformer
    ("classifier", LogisticRegression(max_iter=1000))  # Predictor
])

# ================================
# Step 4: Fit pipeline
# ================================
clf_pipeline.fit(X_train, y_train)

# ================================
# Step 5: Predict outcomes
# ================================
y_pred = clf_pipeline.predict(X_test)

# ================================
# Step 6: Evaluate performance
# ================================
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
