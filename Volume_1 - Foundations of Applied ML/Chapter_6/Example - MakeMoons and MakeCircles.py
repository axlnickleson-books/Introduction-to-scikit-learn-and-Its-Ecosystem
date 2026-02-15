# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 13:02:26 2025

@author: Admin
"""

from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
import numpy as np

# ================================
# Step 1: Generate the datasets
# ================================
X_moons, y_moons = make_moons(
    n_samples=300,
    noise=0.2,
    random_state=42
)

X_circles, y_circles = make_circles(
    n_samples=300,
    noise=0.1,
    factor=0.5,
    random_state=42
)

# Define B&W-friendly markers for binary labels
markers = {0: "o", 1: "s"}     # circle, square
sizes   = {0: 50, 1: 80}
labels  = {0: "Class 0", 1: "Class 1"}

# ================================
# Step 2: Create figure layout
# ================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -------------------------------
# Plot 1: make_moons
# -------------------------------
for cls in np.unique(y_moons):
    axes[0].scatter(
        X_moons[y_moons == cls, 0],
        X_moons[y_moons == cls, 1],
        marker=markers[cls],
        color="black",
        edgecolor="white",
        s=sizes[cls],
        alpha=0.85,
        label=labels[cls]      # <- no cls == 0 condition
    )

axes[0].set_title("make_moons Dataset")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].grid(True, zorder=0)

# -------------------------------
# Plot 2: make_circles
# -------------------------------
for cls in np.unique(y_circles):
    axes[1].scatter(
        X_circles[y_circles == cls, 0],
        X_circles[y_circles == cls, 1],
        marker=markers[cls],
        color="black",
        edgecolor="white",
        s=sizes[cls],
        alpha=0.85
        # no label here
    )

axes[1].set_title("make_circles Dataset")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].grid(True, zorder=0)

# ================================
# Step 3: Shared legend below figure
# ================================
handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.15),
    ncol=2,
    frameon=True,
    shadow=True
)

plt.tight_layout(rect=[0, 0.1, 1, 1])

# ================================
# Step 4: Save and show
# ================================
plt.savefig("make_moons_circles.png", dpi=300, bbox_inches="tight")
plt.show()
