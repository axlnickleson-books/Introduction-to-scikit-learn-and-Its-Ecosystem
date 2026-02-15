# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 03:19:33 2025

@author: Admin
"""

from sklearn.datasets import load_wine
	
# Load the dataset
wine = load_wine()
	
# Inspect the Bunch structure
print(type(wine))
print(wine.keys())
	
# Access elements by key or attribute
X = wine.data
y = wine.target
	
print("Shape of X:", X.shape)
print("Target classes:", wine.target_names)