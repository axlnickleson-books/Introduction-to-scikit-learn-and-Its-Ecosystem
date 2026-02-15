# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 13:13:35 2025

@author: Admin
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

# =========================================
# Generate datasets with different noise levels
# =========================================
X_low, y_low = make_moons(n_samples=300, noise=0.05, random_state=0)
X_high, y_high = make_moons(n_samples=300, noise=0.40, random_state=0)

# Black & white markers
markers = {0: "o", 1: "s"}   # circles & squares
sizes   = {0: 50, 1: 70}

# =========================================
# Create the figure
# =========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- Low-noise plot ----
for cls in np.unique(y_low):
    axes[0].scatter(
        X_low[y_low == cls, 0],
        X_low[y_low == cls, 1],
        c="black",
        marker=markers[cls],
        s=sizes[cls],
        edgecolor="white",
        linewidth=0.5,
        label=f"Class {cls}" if cls == 0 else None
    )

axes[0].set_title("Low Noise (0.05)")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].grid(True, alpha=0.3)

# ---- High-noise plot ----
for cls in np.unique(y_high):
    axes[1].scatter(
        X_high[y_high == cls, 0],
        X_high[y_high == cls, 1],
        c="black",
        marker=markers[cls],
        s=sizes[cls],
        edgecolor="white",
        linewidth=0.5
    )

axes[1].set_title("High Noise (0.40)")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].grid(True, alpha=0.3)

# Global legend centered below both plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, ["Class 0", "Class 1"],
    loc="lower center",
    ncol=2,
    frameon=True
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# =========================================
# Save figure
# =========================================
plt.savefig("moons_noise_effect.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved moons_noise_effect.png with 300 DPI.")
