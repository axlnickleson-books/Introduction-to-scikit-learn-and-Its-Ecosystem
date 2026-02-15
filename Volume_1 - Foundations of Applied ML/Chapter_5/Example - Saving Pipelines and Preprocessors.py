# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:35:04 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Create pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=500))
])

# Train pipeline
pipe.fit(X, y)

# Save entire pipeline
dump(pipe, "logreg_pipeline.joblib")

# Load and predict
loaded_pipe = load("logreg_pipeline.joblib")
print(loaded_pipe.predict(X[:5]))
