# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:30:41 2025

@author: Admin
"""

# Example - Threshold–Metric Curves (e.g., precision vs. threshold)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

# Step 1 — Load dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign (or vice versa depending on dataset)

# Step 2 — Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3 — Train a classifier with probability output
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Step 4 — Get predicted probabilities for the positive class
y_proba = clf.predict_proba(X_test)[:, 1]

# Step 5 — Define a range of thresholds
thresholds = np.linspace(0.0, 1.0, 101)

precision_values = []
recall_values = []

for thr in thresholds:
    # Convert probabilities to hard predictions using the current threshold
    y_pred_thr = (y_proba >= thr).astype(int)

    # Handle edge case: if all predictions are 0, precision is undefined — skip or set to 0
    if y_pred_thr.sum() == 0:
        precision_values.append(0.0)
        recall_values.append(0.0)
    else:
        precision_values.append(precision_score(y_test, y_pred_thr))
        recall_values.append(recall_score(y_test, y_pred_thr))

# Step 6 — Plot precision and recall vs threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision_values, label="Precision", linewidth=2)
plt.plot(thresholds, recall_values, label="Recall", linestyle="--", linewidth=2)
plt.xlabel("Decision Threshold")
plt.ylabel("Metric Value")
plt.title("Precision and Recall vs. Decision Threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_threshold_plot.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()

# Optional: print a small table for a few selected thresholds
selected_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
print("Threshold  Precision  Recall")
for thr in selected_thresholds:
    # Find the closest index in the thresholds array
    idx = (np.abs(thresholds - thr)).argmin()
    print(f"{thresholds[idx]:8.2f}  {precision_values[idx]:9.3f}  {recall_values[idx]:6.3f}")
