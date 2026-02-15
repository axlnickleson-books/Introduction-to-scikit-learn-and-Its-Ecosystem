# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:52:25 2025

@author: Admin
"""


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, auc
)

# Ground truth and predictions
y_true = [0, 0, 1, 1, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0]

# Pretend probabilistic predictions for AUC metrics
y_scores = [0.10, 0.55, 0.91, 0.88, 0.40, 0.20]

# -------------------------------------------------------
# Basic Metrics
# -------------------------------------------------------
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred))
print("MCC:", matthews_corrcoef(y_true, y_pred))

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# -------------------------------------------------------
# Classification Report
# -------------------------------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# -------------------------------------------------------
# ROC AUC (binary classification only)
# -------------------------------------------------------
roc = roc_auc_score(y_true, y_scores)
print("ROC AUC:", roc)

# -------------------------------------------------------
# Precision-Recall AUC
# -------------------------------------------------------
prec, rec, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(rec, prec)
print("PR AUC:", pr_auc)

