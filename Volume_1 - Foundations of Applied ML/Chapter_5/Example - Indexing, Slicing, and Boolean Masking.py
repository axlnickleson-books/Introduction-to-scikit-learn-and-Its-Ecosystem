# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 03:11:47 2025

@author: Admin
"""

import numpy as np 
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
# Standard slicing
print(arr2d[0, 1])    # second element of first row
	
# Row slicing
print(arr2d[1, :])    # entire second row
	
# Column slicing
print(arr2d[:, 2])    # entire third column
	
# Boolean masking
mask = arr2d[:, 1] > 2
print(arr2d[mask])    # select rows with second column > 2