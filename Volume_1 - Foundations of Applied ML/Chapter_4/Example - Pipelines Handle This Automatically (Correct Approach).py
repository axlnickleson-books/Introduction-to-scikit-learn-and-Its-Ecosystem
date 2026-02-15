# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:27:36 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
	
# Load data
data = load_breast_cancer()
X, y = data.data, data.target
	
# Split BEFORE any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42, stratify=y
)
	
# Define pipeline
pipe = Pipeline([
("scaler", StandardScaler()),        # Transformer
("model", LogisticRegression(max_iter=1000))  # Predictor
])
	
# Fit on training data only
pipe.fit(X_train, y_train)
	
# Predict on test data
y_pred = pipe.predict(X_test)
	
# Evaluate
print("Test Accuracy:", accuracy_score(y_test, y_pred))