# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:02:11 2025

@author: Admin
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Step 1 — Load the dataset
# ------------------------------------------------------------
X, y = load_iris(return_X_y=True)

# ------------------------------------------------------------
# Step 2 — Split into training and test sets (stratified)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Step 3 — Initialize and train the Random Forest classifier
# ------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ------------------------------------------------------------
# Step 4 — Predict on the test set
# ------------------------------------------------------------
y_pred = clf.predict(X_test)

# ------------------------------------------------------------
# Step 5 — Evaluate model performance
# ------------------------------------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Step 6 — Display confusion matrix
# ------------------------------------------------------------
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Confusion Matrix – Random Forest (Iris Dataset)")
plt.grid(False)
plt.show()

# ------------------------------------------------------------
# Step 7 — Show feature importances
# ------------------------------------------------------------
print("Feature importances:", clf.feature_importances_)
