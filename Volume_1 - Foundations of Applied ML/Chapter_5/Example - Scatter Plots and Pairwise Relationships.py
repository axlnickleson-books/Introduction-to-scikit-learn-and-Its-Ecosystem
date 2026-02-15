# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 03:48:08 2025

@author: Admin
"""

from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from sklearn
data = load_diabetes()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target variable
df["target"] = data.target

# Simple scatter plot between two real features
plt.figure(figsize=(12,8))
plt.scatter(df["bmi"], df["bp"], alpha=0.6)
plt.xlabel("BMI")
plt.ylabel("Blood Pressure")
plt.title("Scatter Plot of BMI vs. Blood Pressure")
plt.grid(True)
plt.savefig("Scatter_BMI_BP.png",
            dpi = 300, 
            bbox_inches="tight")
plt.show()

# Pairwise relationships across multiple variables
plt.figure(figsize=(12,8))
sns.pairplot(df[["bmi", "bp", "s1", "target"]], hue="target")
plt.savefig("Seaborn_pairplot.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
