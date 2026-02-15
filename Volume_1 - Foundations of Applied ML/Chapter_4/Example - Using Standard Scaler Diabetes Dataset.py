# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 02:49:41 2025

@author: Admin
"""
import numpy as np 
from sklearn.datasets import load_diabetes 
from sklearn.preprocessing import StandardScaler 

data = load_diabetes()
X,y = data.data, data.target 
print("Shape of X:", X.shape)
print("Feature means (first 5):", np.round(X.mean(axis=0)[:5], 2))
print("Feature stds (first 5):", np.round(X.std(axis=0)[:5], 2))

scaler = StandardScaler()
scaler.fit(X)
	
print("Learned means (first 5):", np.round(scaler.mean_[:5], 2))
print("Learned stds (first 5):", np.round(scaler.scale_[:5], 2))

X_scaled = scaler.transform(X)

print("Scaled feature means (first 5):", np.round(X_scaled.mean(axis=0)[:5], 2))
print("Scaled feature stds (first 5):", np.round(X_scaled.std(axis=0)[:5], 2))

scaler = StandardScaler()
X_scaled_alt = scaler.fit_transform(X)
sample_index = 0
print("Original features (first 5):", np.round(X[sample_index, :5], 2))
print("Scaled features (first 5):", np.round(X_scaled[sample_index, :5], 2))














