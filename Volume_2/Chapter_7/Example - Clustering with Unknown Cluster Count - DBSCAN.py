# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:29:34 2025

@author: Admin
"""

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate dataset
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# Apply DBSCAN
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

plt.figure(figsize=(12, 8))

# Main scatter plot (black-and-white friendly)
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,                 # grayscale by cluster
    cmap="gray",              # black & white output
    s=60,
    alpha=0.85,
    edgecolor="black",        # black outlines
    linewidth=0.4,
    zorder=3
)

plt.title("DBSCAN Clustering Example (Black-and-White)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, zorder=0)
plt.tight_layout()
plt.savefig("dbscan_moons_bw.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
