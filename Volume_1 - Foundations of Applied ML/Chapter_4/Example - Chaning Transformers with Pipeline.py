# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 02:41:41 2025

@author: Admin
"""


from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# ================================
# Step 1: Load Titanic dataset
# ================================
print("Loading Titanic dataset...")
titanic = fetch_openml("titanic", version=1, as_frame=True, parser="pandas")
X_full = titanic.data
y = titanic.target
print("Dataset loaded.")
print("Full feature shape:", X_full.shape)
print()

# ================================
# Step 2: Select numeric features
# ================================
print("Selecting numeric features: ['age', 'fare']")
X_num = X_full[["age", "fare"]]
print("Shape of numeric subset:", X_num.shape)
print("Number of missing values per column:")
print(X_num.isna().sum())
print()

# ================================
# Step 3: Build preprocessing pipeline
# ================================
print("Building Pipeline: SimpleImputer + StandardScaler")
preprocess_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
print("Pipeline created.")
print()

# ================================
# Step 4: Fit and transform
# ================================
print("Fitting pipeline and transforming numeric data...")
X_scaled = preprocess_pipeline.fit_transform(X_num)
print("Transformation completed.")
print("Shape after preprocessing:", X_scaled.shape)
print()

# ================================
# Step 5: Inspect results
# ================================
print("First 5 transformed rows:")
print(X_scaled[:5])

