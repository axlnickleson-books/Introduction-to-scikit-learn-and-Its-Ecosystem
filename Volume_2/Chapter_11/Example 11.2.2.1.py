# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 01:40:17 2025

@author: Admin
"""

from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------
# 1. Generate synthetic classification dataset
# ----------------------------------------------------
X, y = make_classification(
    n_samples=500,
    n_features=10,
    random_state=42
)

# ----------------------------------------------------
# 2. Train Gradient Boosting classifier
# ----------------------------------------------------
clf = GradientBoostingClassifier()
clf.fit(X, y)

# ----------------------------------------------------
# 3. Compute permutation importances
# ----------------------------------------------------
results = permutation_importance(
    clf,
    X, y,
    n_repeats=10,
    random_state=42
)

importances = results.importances_mean

# Sort features by importance for cleaner plot
indices = np.argsort(importances)
sorted_importances = importances[indices]
print("importances = ", importances)
print("indices = ", indices)
print("sorted_importances = ", sorted_importances)
# ----------------------------------------------------
# 4. Black-and-White Feature Importance Plot
# ----------------------------------------------------
plt.figure(figsize=(10, 5))
plt.barh(
    range(len(sorted_importances)),
    sorted_importances,
    color="black",
    edgecolor="black",
    zorder=3
)

plt.xlabel("Mean Importance", color="black")
plt.ylabel("Feature Index (sorted)", color="black")
plt.title("Permutation Feature Importance (Black & White)", color="black")

# Make axes and ticks black
ax = plt.gca()
ax.spines["bottom"].set_color("black")
ax.spines["top"].set_color("black")
ax.spines["left"].set_color("black")
ax.spines["right"].set_color("black")

plt.xticks(color="black")
plt.yticks(
    ticks=range(len(sorted_importances)),
    labels=indices,
    color="black"
)

# Add subtle grid behind bars
plt.grid(True, zorder=0)

plt.tight_layout()
plt.savefig("Figure 11.5.png", 
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
