# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:02:59 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    OPTICS
)

# =============================================================================
# 1. Generate Synthetic Dataset
# =============================================================================
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

# =============================================================================
# 2. Define Clustering Models
# =============================================================================
models = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
    "SpectralClustering": SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42),
    "OPTICS": OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
}

# =============================================================================
# 3. Fit and Plot Results for Each Model
# =============================================================================
for name, model in models.items():

    # SpectralClustering needs fit_predict
    if name == "SpectralClustering":
        labels = model.fit_predict(X)
    else:
        model.fit(X)
        labels = model.labels_

    plt.figure(figsize=(12, 8))

    # Main scatter plot
    plt.scatter(
        X[:, 0], X[:, 1],
        c=labels,
        cmap="gray",
        s=50,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
        zorder=2
    )

    # === Make KMeans centroids clearly visible ===
    if name == "KMeans":
        centers = model.cluster_centers_
        plt.scatter(
            centers[:, 0], centers[:, 1],
            s=350,
            c="white",                 # white interior
            edgecolor="black",         # black outline
            linewidth=2,
            marker="X",                # easily visible marker
            label="Centroids",
            zorder=5
        )
        plt.legend()

    plt.title(f"{name} Clustering Example")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig("{}_Clustering_Example.png".format(name),
                dpi = 300, 
                bbox_inches = "tight")
    plt.show()

