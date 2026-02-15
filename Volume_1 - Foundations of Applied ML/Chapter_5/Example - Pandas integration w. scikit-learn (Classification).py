# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 00:11:59 2025

@author: Admin
"""

# Step 1 — Import Libraries
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Step 2 — Load sklearn dataset and convert to pandas DataFrame
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Dataset preview:")
print(df.head())

# Step 3 — Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Step 4 — Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5 — Initialize classifier
clf = RandomForestClassifier(random_state=42)

# Step 6 — Train classifier
clf.fit(X_train, y_train)

# Step 7 — Make predictions
y_pred = clf.predict(X_test)

# Step 8 — Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
