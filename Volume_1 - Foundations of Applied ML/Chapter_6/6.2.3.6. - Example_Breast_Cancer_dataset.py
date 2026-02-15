# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 01:56:28 2025

@author: Admin
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
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
# ---------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------
cancer = load_breast_cancer(as_frame=True)

# Access the DataFrame
df = cancer.frame
print(df.head())

print("Classes:", cancer.target_names)
print("Number of features:", len(cancer.feature_names))
print("Samples:", df.shape[0])

# ---------------------------------------------------------
# Step 2: Prepare features and target for PCA
# ---------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]   # 0 = malignant, 1 = benign

# Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# Step 3: Apply PCA (2 components for visualization)
# ---------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------------------------------------
# Step 4: Create a PCA scatter plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# Separation of classes in black & white
plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    label="Malignant",
    edgecolor="black",
    facecolor="white",
    alpha=0.8,
    s=60
)

plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    label="Benign",
    edgecolor="black",
    facecolor="black",
    alpha=0.8,
    s=60
)

# Labels and aesthetics
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Breast Cancer Dataset")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# ---------------------------------------------------------
# Step 5: Save the figure
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig("breast_cancer_pca.png", dpi=300)
plt.show()
