# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 03:04:27 2025

@author: Admin
"""


import numpy as np 
X = np.array([1, 2, 3, 4, 5])
	
# Reshape into 2D with one feature
X_reshaped = X.reshape(-1, 1)
print("X = {}".format(X))
print("X.shape = {}".format(X.shape))
print("X_reshaped = {}".format(X_reshaped))
print("X_reshaped.shape = {}".format(X_reshaped.shape))   # (5, 1)
