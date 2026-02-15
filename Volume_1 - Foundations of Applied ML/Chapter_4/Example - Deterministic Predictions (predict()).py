# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 03:28:55 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
	
# Load and split dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
	data.data, data.target, test_size=0.3, random_state=42
	)
	
# Train predictor
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
	
# Generate deterministic predictions
y_pred = model.predict(X_test)
print("Predicted class labels:", y_pred[:10])