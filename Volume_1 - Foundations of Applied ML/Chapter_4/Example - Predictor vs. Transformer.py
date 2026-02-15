# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 03:13:28 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
	
# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
	
# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	
# 3. Transformer: scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
	
# 4. Predictor: train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
	
# 5. Predict outcomes
y_pred = model.predict(X_test_scaled)