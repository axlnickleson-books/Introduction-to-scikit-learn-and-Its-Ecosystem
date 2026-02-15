# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 01:16:36 2025

@author: Admin
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# Load the dataset
iris = load_iris()
	
# Features and labels
X = iris.data
y = iris.target
	
print("Feature names:", iris.feature_names)
print("Target classes:", iris.target_names)
print("Shape of X:", X.shape)


# -----------------------------------------------
# Scatter plot: Petal length vs. petal width (BW)
# -----------------------------------------------

# Species names
species_names = iris.target_names

# Black-and-white marker styles
markers = ["o", "s", "^"]   # circle, square, triangle
sizes = [70,140,210]
plt.figure(figsize=(7, 5))

for i, species in enumerate(species_names):
    # Select samples of one species
    subset_x = X[y == i]  # pick rows where label == i
    
    # Plot with unique marker for each class
    plt.scatter(
        subset_x[:, 2],              # petal length (column index 2)
        subset_x[:, 3],              # petal width  (column index 3)
        marker=markers[i],
        facecolor="white",           # ensures grayscale
        edgecolor="black",
        s=sizes[i],
        label=species
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs. Petal Width (Iris Dataset)")
plt.legend(title="Species")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("Figure_6.3_iris_scatter_petal_bw.png", dpi=300,
            bbox_inches = "tight")
plt.show()

print("Saved: Figure_6.3_iris_scatter_petal_bw.png")