# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:50:14 2025

@author: Admin
"""

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import numpy as np

# 1. Load dataset
X, y = load_wine(return_X_y=True)

# 2. Create a stratified K-fold splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Define the model
model = RandomForestClassifier(random_state=42)

# ------------------------------------------------------------
# A) Basic cross-validation with accuracy only
# ------------------------------------------------------------
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("Accuracy per fold:", scores)
print("Mean Accuracy:", scores.mean().round(4))
print("Std  Accuracy:", scores.std().round(4))

# ------------------------------------------------------------
# B) Multi-metric cross-validation (more complete evaluation)
# ------------------------------------------------------------
results = cross_validate(
    model, X, y,
    cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    return_train_score=True
)

print("\n=== Cross-Validation Detailed Metrics ===")
print("Train Accuracy:", np.round(results["train_accuracy"], 4))
print("Test Accuracy :", np.round(results["test_accuracy"], 4))
print("Test Precision:", np.round(results["test_precision_macro"], 4))
print("Test Recall   :", np.round(results["test_recall_macro"], 4))
print("Test F1-score :", np.round(results["test_f1_macro"], 4))

print("\nMean Test Accuracy :", results["test_accuracy"].mean().round(4))
print("Mean Test F1-score :", results["test_f1_macro"].mean().round(4))

# ------------------------------------------------------------
# C) Train the final model and show feature importances
# ------------------------------------------------------------
model.fit(X, y)
importances = model.feature_importances_

print("\nTop 5 Feature Importances:")
sorted_idx = np.argsort(importances)[::-5]  # top 5
for idx in sorted_idx[:5]:
    print(f"{idx:2d} - {importances[idx]:.4f}")