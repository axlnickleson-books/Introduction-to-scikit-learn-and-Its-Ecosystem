# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 00:25:16 2025

@author: Admin
"""

# from sklearn.datasets import load_iris, fetch_openml
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
# accuracy_score, precision_score, recall_score,
# f1_score, roc_auc_score, classification_report
# )

# # --- Toy Dataset ---
# X_toy, y_toy = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(
# X_toy, y_toy, stratify=y_toy, random_state=42
# )

# clf_toy = RandomForestClassifier(random_state=42)
# clf_toy.fit(X_train, y_train)
# y_pred_toy = clf_toy.predict(X_test)

# print("=== Toy Dataset (Iris) ===")
# print("Accuracy :", accuracy_score(y_test, y_pred_toy))
# print("Precision:", precision_score(y_test, y_pred_toy, average='macro'))
# print("Recall   :", recall_score(y_test, y_pred_toy, average='macro'))
# print("F1-score :", f1_score(y_test, y_pred_toy, average='macro'))

# # --- Real-World Dataset ---
# X_real, y_real = fetch_openml("creditcard", version=1, return_X_y=True, as_frame=True)
# Xr_train, Xr_test, yr_train, yr_test = train_test_split(
# X_real, y_real, stratify=y_real, random_state=42
# )

# clf_real = RandomForestClassifier(random_state=42, n_jobs=-1)
# clf_real.fit(Xr_train, yr_train)
# y_pred_real = clf_real.predict(Xr_test)
# y_proba_real = clf_real.predict_proba(Xr_test)[:, 1]

# print("\n=== Real-World Dataset (Credit Card Fraud) ===")
# print("Accuracy :", accuracy_score(yr_test, y_pred_real))
# print("Precision:", precision_score(yr_test, y_pred_real, pos_label='1'))
# print("Recall   :", recall_score(yr_test, y_pred_real, pos_label='1'))
# print("F1-score :", f1_score(yr_test, y_pred_real, pos_label='1'))
# print("ROC-AUC  :", roc_auc_score(yr_test, y_proba_real))


# print("\nDetailed Report for Real-World Dataset:")
# print(classification_report(yr_test, y_pred_real))

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# --- Metric values from your results ---
metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

# Toy dataset (Iris)
toy_scores = [0.9211, 0.9246, 0.9231, 0.9230, np.nan]  # No ROC-AUC for multiclass by default

# Real-world dataset (Credit Card Fraud)
real_scores = [0.9995, 0.9320, 0.7805, 0.8496, 0.9373]

# --- Prepare plot ---
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))

# Black and white bar styles
ax.bar(x - width/2, toy_scores, width, color='white', edgecolor='black', hatch='///', label='Toy (Iris)')
ax.bar(x + width/2, real_scores, width, color='grey', edgecolor='black', hatch='\\\\\\', label='Real-World (Fraud)')

# Add metric labels and formatting
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Toy vs. Real-World Dataset Performance Comparison (B/W)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(frameon=False)

# Grid and minimal layout
ax.grid(axis='y', linestyle='--', linewidth=0.5, color='black', alpha=0.3)
plt.tight_layout()

# --- Save as high-quality PDF for LaTeX ---
plt.savefig("toy_vs_real_accuracy_comparison.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
