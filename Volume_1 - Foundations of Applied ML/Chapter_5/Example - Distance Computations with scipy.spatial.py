# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:31:49 2025

@author: Admin
"""

from scipy.spatial import distance

a = [1, 2, 3]
b = [4, 5, 6]

# Euclidean distance
d_euclidean = distance.euclidean(a, b)

# Cosine distance
d_cosine = distance.cosine(a, b)

print("Euclidean distance:", d_euclidean)
print("Cosine distance:", d_cosine)
