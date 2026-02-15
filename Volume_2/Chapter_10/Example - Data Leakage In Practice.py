# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 02:16:40 2025

@author: Admin
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(423)
# Incorrect: scaling before split (data leakage)
X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # uses entire dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Accuracy (leaked) train", accuracy_score(y_train, model.predict(X_train)))
print("Accuracy (leaked) test:", accuracy_score(y_test, model.predict(X_test)))
	
# Correct: fit scaler only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model2 = RandomForestClassifier()
model2.fit(X_train_scaled, y_train)
print("Accuracy (no leakage) train:", accuracy_score(y_train, model2.predict(X_train_scaled)))
print("Accuracy (no leakage) test:", accuracy_score(y_test, model2.predict(X_test_scaled)))