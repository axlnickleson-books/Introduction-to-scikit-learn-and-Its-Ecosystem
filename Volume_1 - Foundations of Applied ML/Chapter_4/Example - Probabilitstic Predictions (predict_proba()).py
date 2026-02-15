# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 03:40:46 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ================================
# Step 1: Load dataset
# ================================
data = load_breast_cancer()
X, y = data.data, data.target

# ================================
# Step 2: Split dataset
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ================================
# Step 3: Train classifier
# ================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ================================
# Step 4: Predict probabilities
# ================================
y_proba = model.predict_proba(X_test)

print("Predicted probabilities (first 3 samples):")
print(y_proba[:3])
