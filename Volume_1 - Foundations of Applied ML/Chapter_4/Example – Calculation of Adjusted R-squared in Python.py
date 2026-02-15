# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 02:39:16 2025

@author: Admin
"""

import numpy as np
from sklearn.metrics import r2_score

print("=== Adjusted R-squared Calculation in Python ===")

# True values and predictions
y_true = np.array([3, 5, 7, 9])
y_pred = np.array([2.5, 5.5, 6.5, 8.0])

print("y_true:", y_true)
print("y_pred:", y_pred)

# --- Step 1: Compute ordinary R-squared ---
R2 = r2_score(y_true, y_pred)
print("\nR-squared =", R2)

# --- Step 2: Define sample size (n) and number of predictors (p) ---
n = len(y_true)
p = 1   # Example: model has 1 predictor

print("Number of samples (n):", n)
print("Number of predictors (p):", p)

# --- Step 3: Compute adjusted R-squared ---
R2_adj = 1 - ((1 - R2) * (n - 1) / (n - p - 1))
print("\nAdjusted R-squared =", R2_adj)