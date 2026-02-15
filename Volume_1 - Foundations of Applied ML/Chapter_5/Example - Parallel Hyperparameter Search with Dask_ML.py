# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:14:33 2025

@author: Admin
"""

from dask_ml.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from dask.distributed import Client
	
# Start a Dask distributed client
client = Client()
	
# Load dataset
X, y = load_digits(return_X_y=True)
	
# Define model and parameter grid
model = LogisticRegression(max_iter=1000, solver="saga")
param_grid = {"C": [0.1, 1, 10]}
	
# Parallel grid search with Dask
search = GridSearchCV(model, param_grid, cv=5)
search.fit(X, y)
	
print("Best parameter:", search.best_params_)