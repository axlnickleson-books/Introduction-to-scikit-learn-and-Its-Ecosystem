# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 04:05:58 2025

@author: Admin
"""

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
import time

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Split for proper evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(random_state=42)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4]
}

# ---------------------------------------------------------
# Successive Halving
# ---------------------------------------------------------
start = time.time()
halving_grid = HalvingGridSearchCV(
    model, param_grid, cv=5, factor=2, scoring='f1', n_jobs=-1
)
halving_grid.fit(X_train, y_train)
halving_time = time.time() - start

print("Best parameters (Halving):", halving_grid.best_params_)
print("Candidates per iteration:", halving_grid.n_candidates_)
print("Resources per iteration :", halving_grid.n_resources_)

y_pred = halving_grid.best_estimator_.predict(X_test)
print("Test F1 (Halving):", f1_score(y_test, y_pred))
print("Halving runtime:", round(halving_time, 3), "s")

# ---------------------------------------------------------
# Full Grid Search for comparison (optional)
# ---------------------------------------------------------
start = time.time()
full_grid = GridSearchCV(
    model, param_grid, cv=5, scoring='f1', n_jobs=-1
)
full_grid.fit(X_train, y_train)
full_time = time.time() - start

print("\nBest parameters (Grid Search):", full_grid.best_params_)
print("Test F1 (Grid Search):", f1_score(y_test, full_grid.predict(X_test)))
print("Grid Search runtime:", round(full_time, 3), "s")
