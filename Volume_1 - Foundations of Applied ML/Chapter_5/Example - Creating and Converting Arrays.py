# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 02:37:08 2025

@author: Admin
"""

import numpy as np
import pandas as pd
	
# From a Python list
arr = np.array([1, 2, 3, 4, 5])
	
# From a 2D list
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
	
# From a pandas DataFrame
df = pd.DataFrame({"age": [25, 32, 47], "score": [88, 92, 79]})
arr_from_df = df.values   # or df.to_numpy()
print("arr = {}".format(arr))
print("arr.shape = {}".format(arr.shape))
print("arr2d = {}".format(arr2d))
print("arr2d.shape = {}".format(arr2d.shape))
print(df)
print("arr_from_df = {}".format(arr_from_df))
print("arr_from_df.shape = {}".format(arr_from_df.shape))