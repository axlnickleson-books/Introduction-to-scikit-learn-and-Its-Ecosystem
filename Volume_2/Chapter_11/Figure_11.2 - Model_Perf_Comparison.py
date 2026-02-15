# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 01:15:36 2025

@author: Admin
"""

# ============================================================
# Multi-Metric Evaluation Across Classifiers (Black & White)
# Generates: model_diagnostic_metrics.pdf
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
# -------------------- Load dataset --------------------
X, y = load_breast_cancer(return_X_y=True)

# -------------------- Define models --------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
}

# -------------------- Train/test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -------------------- Evaluate models --------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
scores = {name: {} for name in models}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback if no predict_proba
        if hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred

    scores[name]["Accuracy"] = accuracy_score(y_test, y_pred)
    scores[name]["Precision"] = precision_score(y_test, y_pred)
    scores[name]["Recall"] = recall_score(y_test, y_pred)
    scores[name]["F1-score"] = f1_score(y_test, y_pred)
    scores[name]["ROC-AUC"] = roc_auc_score(y_test, y_proba)

# -------------------- Prepare data for plotting --------------------
classifiers = list(models.keys())
vals = np.array([[scores[c][m] for m in metrics] for c in classifiers])

# -------------------- Black & white bar plot --------------------
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(metrics))
width = 0.8 / len(classifiers)
hatches = ['///', '\\\\\\', 'xxx', '---', '...']

for i, clf in enumerate(classifiers):
    ax.bar(
        x + (i - (len(classifiers)-1)/2) * width,
        vals[i],
        width=width,
        color='white',
        edgecolor='black',
        hatch=hatches[i % len(hatches)],
        label=clf
    )

ax.set_ylim(0, 1.05)
ax.set_xlabel("Evaluation Metrics")
ax.set_ylabel("Score")
ax.set_title("Multi-Metric Evaluation Across Classifiers")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
ax.grid(axis='y', linestyle='--', linewidth=0.5, color='black', alpha=0.4)

plt.tight_layout()
plt.savefig("model_diagnostic_metrics.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
