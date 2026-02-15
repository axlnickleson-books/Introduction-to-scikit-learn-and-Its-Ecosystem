# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:17:22 2025

@author: Admin
"""

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load dataset (regression problem)
# ------------------------------------------------------------
X, y = fetch_california_housing(return_X_y=True)

# ------------------------------------------------------------
# Scale features (critical for LASSO)
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# Train–test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42
)

# ------------------------------------------------------------
# Fit a LASSO model
# ------------------------------------------------------------
lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_train, y_train)

print("LASSO coefficients:", lasso.coef_)
print("Number of zero coefficients:", np.sum(lasso.coef_ == 0))

# ------------------------------------------------------------
# Select features using SelectFromModel
# ------------------------------------------------------------
selector = SelectFromModel(lasso, threshold="mean")
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

print("Original shape:", X_train.shape)
print("Selected shape:", X_train_sel.shape)
print("Selected feature indices:", selector.get_support(indices=True))

# ------------------------------------------------------------
# Train a model on selected features
# ------------------------------------------------------------
lasso_sel = Lasso(alpha=0.01, max_iter=5000)
lasso_sel.fit(X_train_sel, y_train)

# ------------------------------------------------------------
# Evaluate performance before and after selection
# ------------------------------------------------------------
y_pred_full = lasso.predict(X_test)
y_pred_sel = lasso_sel.predict(X_test_sel)

print("MSE (full features):", mean_squared_error(y_test, y_pred_full))
print("MSE (selected features):", mean_squared_error(y_test, y_pred_sel))
print("R² (full features):", r2_score(y_test, y_pred_full))
print("R² (selected features):", r2_score(y_test, y_pred_sel))

# ------------------------------------------------------------
# Visualize coefficient magnitudes
# ------------------------------------------------------------
plt.bar(np.arange(len(lasso.coef_)), np.abs(lasso.coef_))
plt.title("LASSO Coefficient Magnitudes (California Housing)")
plt.xlabel("Feature Index")
plt.ylabel("|Coefficient|")
plt.grid(True)
plt.show()
