"""
Performance and training-time comparison of
XGBoost, LightGBM, and CatBoost on the
Breast Cancer dataset (sklearn.datasets).

Requirements:
    pip install xgboost lightgbm catboost
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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

# ---------------------------------------------------------------------
# Step 1: Load dataset and create train/test split
# ---------------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ---------------------------------------------------------------------
# Step 2: Define models (simple, fast configurations)
# ---------------------------------------------------------------------
models = {
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=3,
        verbose=False,
        random_state=42,
    ),
}


# ---------------------------------------------------------------------
# Step 3: Utility function for evaluation
# ---------------------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Fit model, measure training time, and compute metrics."""

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "train_time": train_time,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


# ---------------------------------------------------------------------
# Step 4: Run experiments and collect results
# ---------------------------------------------------------------------
results = {}

for name, model in models.items():
    print(f"Training and evaluating: {name}")
    res = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = res

# Pretty print results
print("\n=== Performance and Training-Time Comparison ===")
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  Training time (s)   : {res['train_time']:.4f}")
    print(f"  Accuracy           : {res['accuracy']:.4f}")
    print(f"  Balanced Accuracy  : {res['balanced_accuracy']:.4f}")
    print(f"  Precision          : {res['precision']:.4f}")
    print(f"  Recall             : {res['recall']:.4f}")
    print(f"  F1-score           : {res['f1']:.4f}")


# ---------------------------------------------------------------------
# Step 5: Black-and-white plot for metrics & training time
# ---------------------------------------------------------------------
model_names = list(results.keys())

accuracy = [results[m]["accuracy"] for m in model_names]
balanced_accuracy = [results[m]["balanced_accuracy"] for m in model_names]
f1_scores = [results[m]["f1"] for m in model_names]
train_times = [results[m]["train_time"] for m in model_names]

# =============================
# Plot 1: Performance Metrics
# =============================

x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(8, 5))

# Bars (black & white)
plt.bar(x - width, accuracy, width, color="black", hatch="////", label="Accuracy")
plt.bar(x, balanced_accuracy, width, color="dimgray", hatch="\\\\\\\\", label="Balanced Accuracy")
plt.bar(x + width, f1_scores, width, color="white", edgecolor="black", hatch="....", label="F1-score")

plt.xticks(x, model_names)
plt.ylim(0.8, 1.01)
plt.ylabel("Score")
plt.title("Performance Metrics (Test Set)")

# Legend BELOW the plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.grid(axis="y", linestyle="--", color="gray", alpha=0.5)
plt.tight_layout()
plt.savefig("XBOO_performance_metrics_test_set.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()


# -------------------- Right: training time --------------------
# =============================
# Plot 2: Training-Time Comparison
# =============================

x = np.arange(len(model_names))
width = 0.4

plt.figure(figsize=(8, 5))

plt.bar(x, train_times, width, color="white", edgecolor="black", hatch="////", label="Training Time (s)")

plt.xticks(x, model_names)
plt.ylabel("Training Time (seconds)")
plt.title("Training-Time Comparison")

# Legend BELOW the plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

plt.grid(axis="y", linestyle="--", color="gray", alpha=0.5)
plt.tight_layout()
plt.savefig("XBOO_training_time_comparison.png",
            dpi = 300, 
            bbox_inches="tight")
plt.show()
