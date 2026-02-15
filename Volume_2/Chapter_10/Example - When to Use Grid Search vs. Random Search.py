# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 04:03:12 2025

@author: Admin
"""

import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(random_state=42)

# -------------------------------------------------------
# Grid Search (exhaustive)
# -------------------------------------------------------
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7]
}

start = time.time()
grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
grid_time = time.time() - start

print("Best Grid Params:", grid.best_params_)
print("Grid Search Best CV Score:", grid.best_score_)
print("Grid Search Test Accuracy:",
      accuracy_score(y_test, grid.best_estimator_.predict(X_test)))
print("Grid Search Runtime:", round(grid_time, 3), "seconds")

# -------------------------------------------------------
# Random Search (sampled combinations)
# -------------------------------------------------------
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(2, 10)
}

start = time.time()
rand = RandomizedSearchCV(
    model, param_dist,
    n_iter=10, cv=5, n_jobs=-1, random_state=42
)
rand.fit(X_train, y_train)
rand_time = time.time() - start

print("\nBest Random Params:", rand.best_params_)
print("Random Search Best CV Score:", rand.best_score_)
print("Random Search Test Accuracy:",
      accuracy_score(y_test, rand.best_estimator_.predict(X_test)))
print("Random Search Runtime:", round(rand_time, 3), "seconds")