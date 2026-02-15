# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 13:21:55 2025

@author: Admin
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_moons, make_circles
import numpy as np

# =========================================================
# Generate datasets
# =========================================================
X_cls, y_cls = make_classification(
    n_samples=400, n_features=2,
    n_redundant=0, n_informative=2,
    class_sep=1.5, random_state=42
)

X_blobs, y_blobs = make_blobs(
    n_samples=400, centers=3,
    cluster_std=1.0, random_state=42
)

X_moons, y_moons = make_moons(
    n_samples=400, noise=0.20, random_state=42
)

X_circles, y_circles = make_circles(
    n_samples=400, noise=0.10, factor=0.5, random_state=42
)

# =========================================================
# Plot configuration
# =========================================================
datasets = [
    (X_cls,     y_cls,     "make_classification"),
    (X_blobs,   y_blobs,   "make_blobs"),
    (X_moons,   y_moons,   "make_moons"),
    (X_circles, y_circles, "make_circles"),
]

# markers for up to 3 classes
markers = ["o", "s", "^"]
sizes = [50, 60, 70]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# =========================================================
# Create subplots in black & white
# =========================================================
for ax, (X, y, title) in zip(axes.ravel(), datasets):

    classes = np.unique(y)

    for idx, cls in enumerate(classes):
        ax.scatter(
            X[y == cls, 0],
            X[y == cls, 1],
            c="black",
            marker=markers[idx % len(markers)],
            s=sizes[idx % len(sizes)],
            edgecolor="white",
            linewidth=0.5,
            label=f"Class {cls}" if idx < 3 else None
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)

# Global title
plt.suptitle("Visual Comparison of Synthetic Datasets")

plt.tight_layout(rect=[0, 0, 1, 0.97])

# =========================================================
# Save to file
# =========================================================
plt.savefig("synthetic_datasets_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved synthetic_datasets_comparison.png (300 DPI black & white).")
