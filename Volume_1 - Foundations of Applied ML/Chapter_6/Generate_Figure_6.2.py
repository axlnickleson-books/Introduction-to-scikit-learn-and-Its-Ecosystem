# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 00:16:20 2025

@author: Admin
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Combine features + target
df = pd.concat([X, y], axis=1)

# Species names
species_names = iris.target_names

# Assign different markers for black-and-white plotting
markers = ["o", "s", "^"]   # circle, square, triangle
linestyles = ["None", "None", "None"]  # No line, dots only
sizes = [70,140,210]
plt.figure(figsize=(12, 8))

for i, species in enumerate(species_names):
    subset = df[df["species"] == i]
    plt.scatter(
        subset["sepal length (cm)"],
        subset["sepal width (cm)"],
        marker=markers[i],
        facecolor="white",
        edgecolor="black",
        s=sizes[i],
        label=species
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs. Sepal Width (Iris Dataset)")
plt.legend(title="Species")
plt.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

# Save figure
plt.tight_layout()
plt.savefig("Figure_6.1_iris_scatter_bw.png", 
dpi=300,
bbox_inches = "tight")
# plt.close()
plt.show()

print("Saved figure as Figure_6.1_iris_scatter_bw.png")
