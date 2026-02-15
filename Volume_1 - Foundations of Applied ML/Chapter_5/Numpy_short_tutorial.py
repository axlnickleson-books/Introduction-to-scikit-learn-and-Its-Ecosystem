# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:24:33 2025

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

X = np.array([1, 2, 3, 4, 5])
	
# Reshape into 2D with one feature
X_reshaped = X.reshape(-1, 1)
print("X = {}".format(X))
print("X.shape = {}".format(X.shape))
print("X_reshaped = {}".format(X_reshaped))
print("X_reshaped.shape = {}".format(X_reshaped.shape))   # (5, 1)


# Standard slicing
print(arr2d[0, 1])    # second element of first row
	
# Row slicing
print(arr2d[1, :])    # entire second row
	
# Column slicing
print(arr2d[:, 2])    # entire third column
	
# Boolean masking
mask = arr2d[:, 1] > 2
print(arr2d[mask])    # select rows with second column > 2
print("#"*72)
a = np.array([1, 2, 3])
b = np.array([10])
	
# Broadcasting: scalar expands to match shape of a
print(a + b)   # [11 12 13]
	
# Broadcasting with different shapes
M = np.ones((3, 3))
v = np.array([1, 2, 3])
print(M + v)

print("#"*72)
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
	
# Matrix multiplication
print(A @ b)
	
# Transpose
print(A.T)
	
# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)
	
# Inverse
A_inv = np.linalg.inv(A)



print("#"*1000)
from scipy.sparse import csr_matrix
	
dense = np.array([[0, 1, 0],
[0, 0, 2],
[3, 0, 0]])
	
sparse = csr_matrix(dense)
print(sparse)


from sklearn.linear_model import LinearRegression
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
	
model = LinearRegression()
model.fit(X, y)
	
print(model.predict(np.array([[5]])))