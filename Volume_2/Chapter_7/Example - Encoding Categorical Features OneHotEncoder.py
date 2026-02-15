# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:48:03 2025

@author: Admin
"""

from sklearn.preprocessing import OneHotEncoder

# Sample categorical dataset
data = [["red"], ["green"], ["blue"], ["green"], ["red"]]

# Initialize the encoder
encoder = OneHotEncoder(sparse_output=False)

# Fit the encoder and transform the data
encoded = encoder.fit_transform(data)

# Inspect the results
print("Encoded matrix:\n", encoded)
print("Categories:", encoder.categories_)
print("Feature names:", encoder.get_feature_names_out())