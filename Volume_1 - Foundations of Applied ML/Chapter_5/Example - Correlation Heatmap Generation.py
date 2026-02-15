# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 09:31:42 2025

@author: Admin
"""

from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1 — Load dataset from scikit-learn
data = load_diabetes()

# Step 2 — Convert the feature matrix into a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Step 3 — Add the target variable (optional but useful for correlation analysis)
df["target"] = data.target

# Step 4 — Compute the correlation matrix
corr = df.corr()

# Step 5 — Plot the correlation heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="Greys", center=0)
plt.title("Correlation Heatmap for the Diabetes Dataset")
plt.tight_layout()
plt.show()


plt.savefig("PearsonsCorrelationHeatmap.png", 
            dpi = 300, 
            bbox_inches = "tight")