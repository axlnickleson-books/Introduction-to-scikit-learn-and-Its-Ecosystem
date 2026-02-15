# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 09:36:55 2025

@author: Admin
"""

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Generate the dataset
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# Step 2: Create a B&W scatter plot with labels
plt.figure(figsize=(12, 8))

# Define marker shapes for each class (B&W friendly)
markers = {0: "o", 1: "s"}
labels = {0: "Class 0", 1: "Class 1"}
s_size = [50, 200]
for cls in np.unique(y):
    plt.scatter(
        X[y == cls, 0],
        X[y == cls, 1],
        marker=markers[cls],
        color="black",
        edgecolor="k",
        s=s_size[cls],
        alpha=0.8,
        label=labels[cls],
        zorder=3
    )

# Annotations
plt.title("Synthetic Binary Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid(True, zorder=0)

# Legend BELOW the plot
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=True,
    shadow=True
)

plt.tight_layout()
plt.savefig("make_classification_example.png",
           dpi = 300,
           bbox_inches = "tight")
plt.show()
