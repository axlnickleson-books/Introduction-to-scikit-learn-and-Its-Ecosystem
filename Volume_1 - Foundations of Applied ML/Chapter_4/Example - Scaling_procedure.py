# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 02:00:16 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
	
# Load dataset and split
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
)
	
# Fit-transform on train, transform on test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
	
print("Feature means after scaling (train):", X_train_scaled.mean(axis=0)[:3])
print("Feature std after scaling (train):", X_train_scaled.std(axis=0)[:3])