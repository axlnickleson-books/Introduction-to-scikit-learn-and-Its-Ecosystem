# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 03:51:52 2025

@author: Admin
"""

# ============================================
# Step 1 — Import Required Libraries
# ============================================
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ============================================
# Step 2 — Load the Dataset
# ============================================
data = load_breast_cancer()
X, y = data.data, data.target

# ============================================
# Step 3 — Split into Train/Test Sets
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================
# Step 4 — Train Logistic Regression
# ============================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ============================================
# Step 5 — Get Predicted Probabilities
# ============================================
y_proba = model.predict_proba(X_test)

# ============================================
# Step 6 — Apply a Custom Decision Threshold
# ============================================
# Extract the probability of class 1 (malignant)
y_proba_class1 = y_proba[:, 1]

# Classify samples as class 1 only if P(class 1) >= 0.7
y_pred_custom = (y_proba_class1 >= 0.7).astype(int)

print("Custom threshold predictions (first 10):")
print(y_pred_custom[:10])
