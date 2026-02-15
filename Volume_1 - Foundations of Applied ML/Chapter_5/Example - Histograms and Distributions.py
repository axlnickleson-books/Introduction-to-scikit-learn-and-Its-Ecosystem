# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 03:20:56 2025

@author: Admin
"""
from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Histogram of a single feature (e.g., "age")
plt.figure(figsize=(12,8))
plt.hist(df["age"], bins=20, color="grey", edgecolor="black", zorder = 3)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.grid(True, zorder = 0)
plt.savefig("Histogram-Matplotlib.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()

plt.figure(figsize=(12,8))
# Using seaborn for a smoother histogram (with KDE)
sns.histplot(df["age"], bins=20, kde=True, color='grey', zorder = 3)
plt.xlabel("Age")
plt.title("Distribution of Age (Seaborn)")
plt.grid(True, zorder = 0 )
plt.savefig("Histogram-Seaborn.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
