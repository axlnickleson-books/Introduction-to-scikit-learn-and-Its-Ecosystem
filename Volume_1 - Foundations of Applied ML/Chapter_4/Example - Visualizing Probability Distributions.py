# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 03:59:25 2025

@author: Admin
"""

# ============================================
# Step 1 — Import Required Libraries
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
# ============================================
# Step 2 — Load the Dataset
# ============================================
data = load_breast_cancer()
X, y = data.data, data.target

# ============================================
# Step 3 — Split into Train/Test Sets
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================
# Step 4 — Train the Predictor
# ============================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ============================================
# Step 5 — Get Predicted Probabilities
# ============================================
y_proba = model.predict_proba(X_test)

# Extract only the probabilities for class 1 (malignant)
y_proba_class1 = y_proba[:, 1]

# ============================================
# Step 6 — Visualize the Probability Distribution
# ============================================
plt.figure(figsize=(12,8))
plt.hist(y_proba_class1, bins=30, color="grey", edgecolor="black",zorder = 3)
plt.title("Distribution of Predicted Probabilities (Class 1)")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.grid(True, zorder = 0)
plt.savefig("Viz_Prob_Dist.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
