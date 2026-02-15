# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 02:28:47 2025

@author: Admin
"""

from sklearn.datasets import load_wine
import pandas as pd
	
# Load the dataset
wine = load_wine()
	
# Convert to DataFrame for easier analysis
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
	
print(df.head())
print("Number of samples:", df.shape[0])
print("Number of features:", df.shape[1] - 1)



