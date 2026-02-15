# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 01:51:20 2025

@author: Admin
"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
	
# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.3, random_state=42)
	
# Create and fit the estimator
model = LinearRegression()
model.fit(X, y)
	
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print(model.get_params())

# Change model configuration dynamically	
model.set_params(fit_intercept=False)
print("Updated parameters:", model.get_params()['fit_intercept'])