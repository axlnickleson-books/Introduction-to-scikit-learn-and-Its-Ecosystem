# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:23:54 2025

@author: Admin
"""

import numpy as np
from scipy import linalg
A = np.array([[1, 2], [3, 4]])
	
# Determinant
det_A = linalg.det(A)
	
# Eigenvalues and eigenvectors
eigvals, eigvecs = linalg.eig(A)
	
# Singular Value Decomposition
U, s, Vh = linalg.svd(A)

print("A = ")