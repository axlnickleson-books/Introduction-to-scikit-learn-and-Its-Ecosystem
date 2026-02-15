# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 01:23:01 2025

@author: Admin
"""

# Step 1 — Import Required Libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 2 — Load sklearn regression dataset and convert to pandas DataFrame
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Dataset preview:")
print(df.head())

# Step 3 — Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 4 — Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5 — Initialize regressor
reg = RandomForestRegressor(random_state=42)

# Step 6 — Train the regressor
reg.fit(X_train, y_train)

# Step 7 — Make predictions
y_pred = reg.predict(X_test)

# Step 8 — Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
