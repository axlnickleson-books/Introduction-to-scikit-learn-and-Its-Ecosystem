# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 00:47:37 2025

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------
# 1. Load dataset
# ---------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# ---------------------------------------
# 2. Apply RFE with Random Forest
# ---------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)

selector = RFE(
    estimator=rf,
    n_features_to_select=10
)

selector.fit(X, y)

# ---------------------------------------
# 3. Retrieve selected features
# ---------------------------------------
selected_mask = selector.support_
selected_features = feature_names[selected_mask]

print("Selected features:", selected_features)

# ---------------------------------------
# 4. Fit RF on selected features for importances
# ---------------------------------------
rf.fit(X[:, selected_mask], y)
importances = rf.feature_importances_

# ---------------------------------------
# 5. Black & White Feature Importance Plot
# ---------------------------------------
plt.figure(figsize=(10, 5))
plt.barh(
    selected_features,
    importances,
    color="black",     # <-- black bars
    edgecolor="black",  # <-- crisp edges
    zorder = 3
)

plt.xlabel("Feature Importance", color="black")
plt.ylabel("Selected Features", color="black")
plt.title("Top 10 Selected Features (RFE + Random Forest)", color="black")

# Make all frame & ticks black
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['top'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')

plt.xticks(color='black')
plt.yticks(color='black')
plt.grid(True,zorder = 0)

plt.tight_layout()
plt.savefig("Figure 11.4.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
