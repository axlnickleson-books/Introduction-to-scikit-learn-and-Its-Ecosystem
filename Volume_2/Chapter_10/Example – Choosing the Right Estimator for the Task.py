# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:47:27 2025

@author: Admin
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np

# 1. Load dataset and split
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y
)

# 2. Inspect class balance
unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution (train):")
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls}: {cnt} samples")

# 3. Define baseline and real model
baseline = DummyClassifier(strategy="most_frequent")
logreg   = LogisticRegression(max_iter=1000)

baseline.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# 4. Predictions
y_pred_baseline = baseline.predict(X_test)
y_pred_logreg   = logreg.predict(X_test)

# 5. Helper for printing metrics
def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

print_metrics("Baseline (most frequent)", y_test, y_pred_baseline)
print_metrics("Logistic Regression",     y_test, y_pred_logreg)