# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:06:23 2025

@author: Admin
"""

# Step 1 — Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 2 — Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 3 — Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4 — Train a simple classifier
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)

# Step 5 — Predict on test data
y_pred = clf.predict(X_test)

# Step 6 — Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 7 — Plot confusion matrix heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greys",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("ConfusionMatrix_Breast_Example.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
