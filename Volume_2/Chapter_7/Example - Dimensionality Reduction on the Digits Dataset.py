# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:45:30 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =============================================================================
# 1. Load dataset
# =============================================================================
X, y = load_digits(return_X_y=True)

# =============================================================================
# 2. Dimensionality Reduction Models (4× grid)
# =============================================================================
embedders = {
    "PCA": PCA(n_components=2, random_state=42),
    "Truncated SVD": TruncatedSVD(n_components=2, random_state=42),
    "t-SNE": TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    ),
    "LDA": LinearDiscriminantAnalysis(n_components=2)
}

# =============================================================================
# 3. Compute embeddings
# =============================================================================
embeddings = {}

for name, model in embedders.items():
    print(f"Computing {name}...")
    if isinstance(model, LinearDiscriminantAnalysis):
        X_2d = model.fit_transform(X, y)
    else:
        X_2d = model.fit_transform(X)
    embeddings[name] = X_2d

# =============================================================================
# 4. Plot 2×2 grid (black-and-white friendly)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

last_scatter = None

for ax, (name, X_2d) in zip(axes, embeddings.items()):
    sc = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        cmap="gray",      # grayscale colormap
        s=10,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.2,
        zorder = 3
    )
    last_scatter = sc

    ax.set_title(name)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True,zorder = 0)
    # show ticks
    ax.tick_params(
        axis='both',
        which='both',
        labelsize=9,
        direction='out'
    )

fig.suptitle("Dimensionality Reduction on Digits Dataset (Black-and-White)")

# =============================================================================
# 5. Clean external colorbar
# =============================================================================

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(last_scatter, cax=cbar_ax)
cbar.set_label("Digit label")
plt.savefig("digits_dimensionality_reduction_bw.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()

