# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:27:30 2025

@author: Admin
"""

# Step 1 — Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Step 2 — Load dataset
data = load_breast_cancer()
X = data.data
y = data.target   # 0 = malignant, 1 = benign

# Step 3 — Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4 — Train a classifier
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)

# Step 5 — Get predicted probabilities for the positive class
# predict_proba returns [P(class 0), P(class 1)] for each sample
y_proba = clf.predict_proba(X_test)[:, 1]

# Step 6 — Compute ROC curve points and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Step 7 — Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", linewidth=2)

# random-guess baseline
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (Breast Cancer)")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("ROC_Curve_BreastCancer.png", 
            dpi = 300, 
            bbox_inches="tight")
plt.show()
