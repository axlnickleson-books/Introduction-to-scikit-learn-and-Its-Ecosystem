# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:50:13 2025

@author: Admin
"""


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Sample dataset with numeric and categorical features
# Columns:
#   0 = Age (numeric)
#   1 = Income (numeric)
#   2 = Color preference (categorical)
X = np.array([
    [25, 50000, "red"],
    [32, np.nan, "blue"],
    [47, 82000, "green"],
    [29, 61000, "red"]
], dtype=object)

# Identify numeric and categorical columns
numeric_features = [0, 1]
categorical_features = [2]

# Define transformers for each column type
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])

# Combine transformations using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Build a full preprocessing pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

# Fit the pipeline and transform the data
X_processed = pipeline.fit_transform(X)

# Inspect the results
print("Processed data:\n", X_processed)
print("Numeric means:", pipeline.named_steps["preprocessor"]
      .named_transformers_["num"]
      .named_steps["imputer"].statistics_)
print("Categorical categories:",
      pipeline.named_steps["preprocessor"]
      .named_transformers_["cat"]
      .named_steps["encoder"].categories_)

