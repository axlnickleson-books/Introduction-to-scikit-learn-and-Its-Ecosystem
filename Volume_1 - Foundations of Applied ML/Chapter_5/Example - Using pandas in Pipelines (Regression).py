# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 02:19:14 2025

@author: Admin
"""

# Step 1 — Import required libraries
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

# Step 2 — Load the Ames Housing (house_prices) dataset from OpenML
house = fetch_openml("house_prices", as_frame=True)
df = house.frame

print("Dataset preview:")
print(df.head())

# Target is SalePrice
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Step 3 — Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# print("\nNumber of numeric features:", len(numeric_features))
# print("Number of categorical features:", len(categorical_features))

# Step 4 — Define preprocessing for numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Step 5 — Build the Random Forest regression pipeline
reg = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

# Step 6 — Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7 — Train the model
reg.fit(X_train, y_train)

# Step 8 — Make predictions
y_pred = reg.predict(X_test)

# Step 9 — Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)
