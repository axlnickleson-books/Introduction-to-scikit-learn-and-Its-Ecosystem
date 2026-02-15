# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 02:26:27 2025

@author: Admin
"""

from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
diabetes = load_diabetes(as_frame=True)

# Access the DataFrame
df = diabetes.frame
print(df.head())

# Display target description
print("Target variable:", diabetes.target[:5])
print("Number of samples:", df.shape[0])
print("Number of features:", len(diabetes.feature_names))

# Visualize relationship between BMI and disease progression
plt.figure(figsize=(8,8))
plt.scatter(df["bmi"], df["target"], color="black")
plt.xlabel("Body Mass Index (BMI)")
plt.ylabel("Disease Progression")
plt.title("Relationship between BMI and Diabetes Progression")
plt.grid(True)
plt.savefig("Diabetes_disease_Vs_BMI.png",
            dpi = 300,
            bbox_inches="tight")

plt.show()