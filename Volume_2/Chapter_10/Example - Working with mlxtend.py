# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 01:57:55 2025

@author: Admin
"""

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from mlxtend.classifier import StackingClassifier

# === Load dataset ===
X, y = load_breast_cancer(return_X_y=True)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Define base learners ===
clf_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

clf_gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    random_state=42
)

# === Define meta-learner ===
meta_clf = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# === Optional: wrap tree models in pipelines with scaling ===
# (not required for trees, but shown here for consistency)
pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", clf_rf),
])

pipe_gb = Pipeline([
    ("scaler", StandardScaler()),
    ("model", clf_gb),
])

# === Define StackingClassifier from mlxtend ===
stack_clf = StackingClassifier(
    classifiers=[pipe_rf, pipe_gb],
    meta_classifier=meta_clf,
    use_probas=True
)

# === Fit stacking classifier ===
stack_clf.fit(X_train, y_train)

# === Evaluate on the test set ===
test_acc = stack_clf.score(X_test, y_test)
print(f"Test accuracy (StackingClassifier): {test_acc:.4f}")

# === Optional: cross-validation on the training data ===
cv_scores = cross_val_score(
    stack_clf,
    X_train,
    y_train,
    cv=5,
    n_jobs=-1
)
print("CV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())