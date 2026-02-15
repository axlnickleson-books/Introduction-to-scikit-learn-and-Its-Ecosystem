# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 01:51:33 2025

@author: Admin
"""

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
	
# Load dataset
data = load_breast_cancer()
X = data.data
	
# Initialize and fit transformer
scaler = StandardScaler()
scaler.fit(X)
	
# Apply transformation
X_scaled = scaler.transform(X)
	
print("Original mean:", X.mean(axis=0)[:3])
print("Transformed mean:", X_scaled.mean(axis=0)[:3])
print("Transformed std deviation:", X_scaled.std(axis=0)[:3])