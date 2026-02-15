# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:52:18 2025

@author: Admin
"""

# Step 1 — Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # you can swap this for any classifier

# Step 2 — Generate a toy 2D dataset
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# Step 3 — Train/test split (optional, but common)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4 — Train a classifier on the two features
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Step 5 — Create a mesh grid over the feature space
h = 0.02  # step size in the mesh
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# Step 6 — Predict class for each point in the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid_points)
Z = Z.reshape(xx.shape)

# Step 7 — Plot decision boundary and training points
plt.figure(figsize=(8, 6))

# Filled contour for decision regions
plt.contourf(xx, yy, Z, alpha=0.3)

# Training points
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolor="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary of a Classifier")
plt.tight_layout()
plt.savefig("Decision_Boundary_Classifier.png",
            dpi = 300,
            bbox_inches= "tight")
plt.show()
