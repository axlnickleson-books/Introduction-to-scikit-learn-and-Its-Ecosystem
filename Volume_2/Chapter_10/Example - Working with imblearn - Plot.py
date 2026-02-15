# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 00:39:14 2025

@author: Admin
"""

import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
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

# === Load dataset ===
data = load_breast_cancer()
X = data.data
y = data.target

# Original distribution
orig_counts = Counter(y)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

train_before_counts = Counter(y_train)

# === Scale + SMOTE just on training ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
train_after_counts = Counter(y_res)

# === Prepare data for plotting ===
labels = ["Class 0", "Class 1"]
orig_vals = [orig_counts[0], orig_counts[1]]
train_before_vals = [train_before_counts[0], train_before_counts[1]]
train_after_vals = [train_after_counts[0], train_after_counts[1]]

x = range(len(labels))
width = 0.25  # bar width

# === Create black & white plot ===
plt.figure(figsize=(12, 8))

plt.bar(
    [p - width for p in x], orig_vals, width=width,
    color="white", edgecolor="black", hatch="///", label="Original", 
    zorder = 3
)
for i in range(len(x)):
    plt.text(x[i] - width-0.025, orig_vals[i]+5, str(orig_vals[i]) )
plt.bar(
    x, train_before_vals, width=width,
    color="white", edgecolor="black", hatch="\\\\\\", label="Train Before SMOTE", 
    zorder = 3
)
for i in range(len(x)):
    plt.text(x[i]-0.025, train_before_vals[i]+5, str(train_before_vals[i]) )
plt.bar(
    [p + width for p in x], train_after_vals, width=width,
    color="white", edgecolor="black", hatch="...", label="Train After SMOTE", 
    zorder = 3
)
for i in range(len(x)):
    plt.text(x[i] + width-0.025, train_after_vals[i]+5, str(train_after_vals[i]) )

# === Add labels ===
plt.xticks(x, labels)
plt.ylabel("Count")
plt.title("Class Distribution Before and After SMOTE (Black & White)", fontsize=28)
plt.legend()
plt.tight_layout()
plt.ylim(0,400)
plt.grid(True, zorder = 0)
plt.savefig("Class Distribution Before and After Smote.png",
            dpi = 300,
            bbox_inches = "tight")
plt.show()
