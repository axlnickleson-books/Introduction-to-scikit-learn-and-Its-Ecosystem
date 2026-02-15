# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:27:06 2025

@author: Admin
"""

# Step 1 — Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

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

# Step 5 — Predicted probabilities for the positive class
y_proba = clf.predict_proba(X_test)[:, 1]

# Step 6 — Compute Precision–Recall curve values
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)

# Step 7 — Plot Precision–Recall Curve
plt.figure(figsize=(8, 6))

plt.plot(recall, precision, label=f"PR Curve (AP = {avg_prec:.2f})", linewidth=2)

# random baseline = proportion of positive class in the data
positive_rate = np.mean(y_test)
plt.hlines(positive_rate, xmin=0, xmax=1, linestyle="--", color="gray",
           label=f"Baseline (pos rate = {positive_rate:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve - Logistic Regression (Breast Cancer)")
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Precision_Recall_Curve_BreastCancer.png",
            dpi = 300, 
            bbox_inches="tight")
plt.show()
