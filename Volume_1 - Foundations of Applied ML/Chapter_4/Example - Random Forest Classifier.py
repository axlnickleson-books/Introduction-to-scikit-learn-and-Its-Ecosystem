# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 04:11:37 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

#split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=42, stratify=y
	)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Initialize the model
rf_clf = RandomForestClassifier(
n_estimators=200,    # number of trees
max_depth=None,      # allow trees to grow fully
random_state=42,     # reproducibility
n_jobs=-1            # use all CPU cores
)
	
# Train (fit) the model
rf_clf.fit(X_train, y_train)
# Predict class labels
red = rf_clf.predict(X_test)

# Predict class labels
y_pred = rf_clf.predict(X_test)
	
# Predict probabilities (optional)
y_proba = rf_clf.predict_proba(X_test)
	
print("Predicted labels (first 10):", y_pred[:10])
print("Predicted probabilities (first 3 samples):")
print(y_proba[:3])


# Compute accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
	
# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

import numpy as np

# Sort and display feature importances
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
	
print("Top 5 important features:")
for i in range(5):
    print(f"{data.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")