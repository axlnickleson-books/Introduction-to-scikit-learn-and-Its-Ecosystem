# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 02:40:35 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

data = load_breast_cancer()
X, y = data.data, data.target  # X: (n_samples, n_features), y: (n_samples,)


X_train, X_test, y_train, y_test = train_test_split(
	X, y,
	test_size=0.30,
	random_state=42,
	stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train, transform train
X_test_scaled  = scaler.transform(X_test)       # transform test with train 

clf = LogisticRegression(
	penalty="l2",        # default L2 regularization
	solver="lbfgs",      # robust for smaller/medium problems
	max_iter=1000,       # increase to ensure convergence
	n_jobs=None,         # single-threaded (set to -1 for all cores if supported)
	random_state=42
	)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)              # hard labels {0,1}
y_proba = clf.predict_proba(X_test_scaled)[:, 1] # P(y=1 | X), needed for ROC-AUC


acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm  = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data.target_names)
	
print(f"Accuracy  : {acc:.4f}")
print(f"ROC AUC   : {auc:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

import numpy as np
coef = clf.coef_.ravel()                 # shape: (n_features,)
feature_names = data.feature_names
top_idx = np.argsort(np.abs(coef))[::-1][:10]  # top 10 by absolute magnitude
	
print("Top 10 influential features (by |coefficient|):")
for i in top_idx:
	sign = "positive" if coef[i] >= 0 else "negative"
	print(f"{feature_names[i]:<30s} {coef[i]:> .4f} ({sign})")

