# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:06:42 2025

@author: Admin
"""

import numpy as np
from scipy import linalg

A = np.array([[1, 2],
              [3, 4]])

# Determinant
det_A = linalg.det(A)

# Eigenvalues and eigenvectors
eigvals, eigvecs = linalg.eig(A)

# Singular Value Decomposition
U, s, Vh = linalg.svd(A)

print("Matrix A:\n", A)
print("Determinant of A:", det_A)
print("\nEigenvalues:\n", eigvals)
print("\nEigenvectors:\n", eigvecs)
print("\nU matrix from SVD:\n", U)
print("Singular values:", s)
print("Vh matrix from SVD:\n", Vh)
