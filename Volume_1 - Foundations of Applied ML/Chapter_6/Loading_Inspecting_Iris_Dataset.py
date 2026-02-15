# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 03:01:11 2025

@author: Admin
"""

from sklearn.datasets import load_iris
	
# Step 1: Load the dataset
iris = load_iris()
	
# Step 2: Extract features and target
X = iris.data
y = iris.target
	
# Step 3: Inspect shapes and metadata
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print("Feature names:", iris.feature_names)
print("Target classes:", iris.target_names)

print(iris.DESCR[:1000])