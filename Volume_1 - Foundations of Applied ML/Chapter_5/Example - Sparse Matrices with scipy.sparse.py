# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:30:09 2025

@author: Admin
"""
import numpy as np 
from scipy.sparse import csr_matrix
	
dense = np.array([[0, 0, 1],
[1, 0, 0],
[0, 2, 0]])
	
# Convert to sparse format
sparse = csr_matrix(dense)
print(sparse)