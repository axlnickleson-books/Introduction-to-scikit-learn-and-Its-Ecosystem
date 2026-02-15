# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:17:22 2025

@author: Admin
"""

from scipy.optimize import minimize
	
# Define a simple quadratic function
f = lambda x: (x - 3)**2 + 4
	
# Minimize starting from x=0
result = minimize(f, x0=0)
print("Optimal value:", result.x)