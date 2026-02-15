# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 01:45:04 2025

@author: Admin
"""

import statsmodels.api as sm
import numpy as np

# Example data
np.random.seed(42)
X = np.random.rand(100, 1) * 10     # single feature
y = 3 * X.squeeze() + 5 + np.random.randn(100) * 2  # linear relation with noise

# Add intercept term
X_with_const = sm.add_constant(X)

# Fit OLS regression
model = sm.OLS(y, X_with_const).fit()

# Print model summary
print(model.summary())
