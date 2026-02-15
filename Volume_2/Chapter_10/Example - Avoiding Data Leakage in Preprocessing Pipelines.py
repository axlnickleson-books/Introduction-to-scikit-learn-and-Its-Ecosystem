# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:32:10 2025

@author: Admin
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Correct scaling: fit only on the training set
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit only on training data
X_test_scaled = scaler.transform(X_test)        # apply to test data

# Train a model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy (no leakage):", accuracy_score(y_test, y_pred))
