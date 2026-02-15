# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:56:06 2025

@author: Admin
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# --------------------------------------------------------
# 1. Load dataset
# --------------------------------------------------------
X, y = load_iris(return_X_y=True)

# --------------------------------------------------------
# 2. Split into training and test sets
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --------------------------------------------------------
# 3. Compare shallow, medium, and deep trees
#    to demonstrate underfitting and overfitting
# --------------------------------------------------------
depths = {
    1:  "Underfitting (very shallow tree)",
    3:  "Good balance",
    10: "Overfitting (very deep tree)"
}

for depth, label in depths.items():
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    print(f"{label:35s} | Depth={depth:2d} | "
          f"Train={train_acc:.3f} | Test={test_acc:.3f}")