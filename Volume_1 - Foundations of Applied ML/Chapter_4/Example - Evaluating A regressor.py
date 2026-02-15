# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 08:30:00 2025

@author: Admin

Example - Evaluating a Regressor (RandomForestRegressor on California Housing)
"""

# ============================================================
# Imports
# ============================================================
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    max_error,
    mean_poisson_deviance,
    mean_gamma_deviance
)

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Matplotlib configuration (Times New Roman + bigger fonts)
# ------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # y tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend
plt.rc('figure', titlesize=BIGGER_SIZE)  # figure title

# ============================================================
# Load dataset
# ============================================================
data = fetch_california_housing()
X, y = data.data, data.target


# ============================================================
# Train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# Initialize and train the regressor
# ============================================================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ============================================================
# Make predictions
# ============================================================
y_pred = model.predict(X_test)

print("\nFirst 10 true values:     ", y_test[:10])
print("First 10 predicted values:", y_pred[:10])

# ============================================================
# Compute regression evaluation metrics
# ============================================================
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)

# For deviance metrics, ensure positivity (California Housing is positive)
# If you want to be extra safe, you can clip predictions to a small positive value.
y_test_pos = y_test
y_pred_pos = np.clip(y_pred, 1e-9, None)

poisson_dev = mean_poisson_deviance(y_test_pos, y_pred_pos)
gamma_dev   = mean_gamma_deviance(y_test_pos, y_pred_pos)

# ============================================================
# Print metric results
# ============================================================
print("\n=== Regression Metrics (Test Set) ===")
print(f"MAE              : {mae:.4f}")
print(f"MSE              : {mse:.4f}")
print(f"RMSE             : {rmse:.4f}")
print(f"MAPE             : {mape:.4f}")
print(f"R² Score         : {r2:.4f}")
print(f"Max Error        : {max_err:.4f}")
print(f"Poisson Deviance : {poisson_dev:.4f}")
print(f"Gamma Deviance   : {gamma_dev:.4f}")

# ============================================================
# Scatter plot: True vs Predicted
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    linewidth=2
)

ax.set_xlabel("True Values (Median House Value)")
ax.set_ylabel("Predicted Values (Median House Value)")
ax.set_title("True vs Predicted Values — Random Forest Regressor")
ax.grid(True)

plt.savefig(
    "rf_regressor_true_vs_pred.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# Train vs Test R²
# ============================================================
train_r2 = model.score(X_train, y_train)
test_r2  = model.score(X_test, y_test)

print("\n=== Train vs Test R² ===")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R² : {test_r2:.4f}")
