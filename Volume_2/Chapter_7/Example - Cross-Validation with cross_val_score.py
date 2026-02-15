# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:50:16 2025

@author: Admin
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Initialize the model
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# Perform 5-fold cross-validation
scores = cross_val_score(
    estimator=clf,
    X=X,
    y=y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Print results
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())
print("Standard deviation:", scores.std())