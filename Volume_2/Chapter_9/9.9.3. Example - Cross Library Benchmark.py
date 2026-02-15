# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 03:47:49 2026

@author: Admin
"""

import time
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
X, y = make_classification(
    n_samples=50000,
    n_features=30,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)
models = [
    ("LogisticRegression", LogisticRegression(max_iter=500)),
    ("SVC", SVC()),
    ("XGBoost", XGBClassifier(eval_metric="logloss", use_label_encoder=False)),
    ("LightGBM", LGBMClassifier()),
    ("CatBoost", CatBoostClassifier(verbose=0))
]
results = []

for name, model in models:
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    acc = accuracy_score(y_test, model.predict(X_test))
    results.append((name, acc, train_time))

df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Training_Time(s)"]
)

print(df.sort_values(by="Accuracy", ascending=False))
data = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

import matplotlib.pyplot as plt
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
# Data
models = ["LightGBM", "CatBoost", "XGBoost", "SVC", "LogisticRegression"]
accuracy = [data['Accuracy'][i] for i in range(len(models))]
training_time = [data['Training_Time(s)'][i] for i in range(len(models))]
offsets = {
    "XGBoost": (10, -5),
    "LightGBM": (10, 10),
    "CatBoost": (10, 10),
    "SVC": (10, 10),
    "LogisticRegression": (10, 10),
}
# Plot
plt.figure(figsize=(12, 8))
plt.scatter(training_time, accuracy)

# Annotate points
for _, r in data.iterrows():
    name = r["Model"]
    dx, dy = offsets.get(name, (8, 8))
    plt.annotate(
        name,
        (r["Training_Time(s)"], r["Accuracy"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="right",
        va="bottom"
    )

plt.xlabel("Training Time (s)")
plt.ylabel("Accuracy")
plt.title("Cross-library benchmark: Training time vs. accuracy")
plt.xlim(-20, 80)
plt.ylim(0.9, 1.0)
plt.grid(True)
plt.tight_layout()

# Save for LaTeX
plt.savefig("cross_library_benchmark.png", dpi=300)
plt.show()