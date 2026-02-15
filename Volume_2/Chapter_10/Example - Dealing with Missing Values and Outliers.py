# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:25:24 2025

@author: Admin
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# ============================================================
# 1. Load the dataset as a DataFrame
# ============================================================
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target

# ============================================================
# 2. Introduce artificial missing values (5% missing)
# ============================================================
rng = np.random.default_rng(42)
mask = rng.uniform(0, 1, size=X.shape) < 0.05
X_missing = X.mask(mask)

print("Number of missing values:", X_missing.isna().sum().sum())

# ============================================================
# 3. Introduce artificial outliers
#    Multiply a few random rows by a large factor
# ============================================================
outlier_indices = rng.choice(len(X), size=5, replace=False)
X_missing.iloc[outlier_indices] *= 10

print("Outlier rows:", outlier_indices)

# ============================================================
# 4. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_missing, y, test_size=0.2, random_state=42
)

# ============================================================
# 5. Impute missing values (mean strategy)
# ============================================================
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# ============================================================
# 6. Scale using RobustScaler (resistant to outliers)
# ============================================================
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("Transformed training data (first 5 rows):")
print(X_train_scaled[:5])