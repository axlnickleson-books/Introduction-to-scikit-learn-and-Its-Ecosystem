# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 02:00:16 2025

@author: Admin
"""

from sklearn.datasets import fetch_openml
	
# Step 1: Load a dataset by name
adult = fetch_openml(name="adult", version=2, as_frame=True)
	
# Step 2: Inspect structure
print("Shape:", adult.data.shape)
print("Features:", adult.feature_names[:5])
print("Target column:", adult.target.name)
	
# Step 3: View the first few samples
print(adult.frame.head())