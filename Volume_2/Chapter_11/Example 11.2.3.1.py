# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 02:16:27 2025

@author: Admin
"""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.datasets import load_wine
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
# --------------------------------------------------
# 1. Load dataset and create train/test split
# --------------------------------------------------
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)

# --------------------------------------------------
# 2. Define base estimators and stacking classifier
# --------------------------------------------------
estimators = [
    ("dt", DecisionTreeClassifier(max_depth=5)),
    ("svm", SVC(probability=True)),
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)

# --------------------------------------------------
# 3. Cross-validation with multiple metrics
# --------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}

cv_results = cross_validate(
    stack,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
)

print("=== Cross-Validation Results (StackingClassifier) ===")
for metric_name in scoring.keys():
    scores = cv_results[f"test_{metric_name}"]
    print(
        f"{metric_name:>16}: "
        f"mean = {scores.mean():.4f}, std = {scores.std():.4f}, "
        f"scores = {np.round(scores, 4)}"
    )
print()

# --------------------------------------------------
# 4. Fit on train set and evaluate on test set
# --------------------------------------------------
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("=== Test Set Evaluation (StackingClassifier) ===")
print(f"Accuracy        : {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro)   : {rec:.4f}")
print(f"F1-score (macro) : {f1:.4f}")
print()

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("=== Confusion Matrix ===")
print(cm)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='gray_r')  # <-- grayscale
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(cm))
plt.xticks(tick_marks, range(len(cm)))
plt.yticks(tick_marks, range(len(cm)))

# Add labels inside the cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.tight_layout()
plt.savefig("Figure_11.9.png", 
            dpi = 300, 
            bbox_inches = "tight")
plt.show()