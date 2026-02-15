# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 04:15:18 2025

@author: Admin
"""

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# ============================================
# 1. Load dataset and train/test split
# ============================================
X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Common parameter distribution for RandomizedSearchCV
param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(2, 10),
}

# ============================================
# 2. Baseline model (no hyperparameter search)
# ============================================
baseline_model = RandomForestClassifier(random_state=42)

start = time.time()
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - start

baseline_test_acc = accuracy_score(y_test, baseline_model.predict(X_test))

print("=== Baseline (no search) ===")
print(f"Training time: {baseline_time:.2f} s")
print(f"Test accuracy: {baseline_test_acc:.4f}")

# ============================================
# 3. Random Search with FEWER iterations (fast, coarse)
# ============================================
start = time.time()
rand_search_small = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5,          # few iterations
    cv=5,
    n_jobs=-1,
    random_state=42
)
rand_search_small.fit(X_train, y_train)
time_small = time.time() - start

best_small = rand_search_small.best_estimator_
small_test_acc = accuracy_score(y_test, best_small.predict(X_test))

print("\n=== RandomizedSearchCV (n_iter=5) ===")
print("Best params:", rand_search_small.best_params_)
print(f"Best CV score: {rand_search_small.best_score_:.4f}")
print(f"Test accuracy: {small_test_acc:.4f}")
print(f"Search time:   {time_small:.2f} s")

# ============================================
# 4. Random Search with MORE iterations (slower, finer)
# ============================================
start = time.time()
rand_search_large = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=30,         # more iterations
    cv=5,
    n_jobs=-1,
    random_state=42
)
rand_search_large.fit(X_train, y_train)
time_large = time.time() - start

best_large = rand_search_large.best_estimator_
large_test_acc = accuracy_score(y_test, best_large.predict(X_test))

print("\n=== RandomizedSearchCV (n_iter=30) ===")
print("Best params:", rand_search_large.best_params_)
print(f"Best CV score: {rand_search_large.best_score_:.4f}")
print(f"Test accuracy: {large_test_acc:.4f}")
print(f"Search time:   {time_large:.2f} s")
