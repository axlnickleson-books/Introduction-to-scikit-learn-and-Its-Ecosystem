from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
f1_score, confusion_matrix, classification_report,
roc_auc_score, RocCurveDisplay)
	
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = load_breast_cancer()
X, y = data.data, data.target
	
print("Shape X, y:", X.shape, y.shape)
print("Target classes:", data.target_names)   # ['malignant' 'benign']
print("First 5 features:", list(data.feature_names[:5]))
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
df['target_label'] = df['target'].map({0: 'malignant', 1: 'benign'})

# Basic stats (count, mean, std, min, quartiles, max)
# Basic stats (count, mean, std, min, quartiles, max)
desc = df[data.feature_names].describe().T
desc.to_csv("Brest_cancer_descriptive_statistics.csv")

class_counts = df['target_label'].value_counts()
class_ratio = class_counts / len(df)
print("Class counts:\n", class_counts)
print("Class ratio:\n", class_ratio)

corr_with_target = df[data.feature_names].corrwith(df['target']).sort_values(ascending=False)
print("Top 10 positively correlated with benign (y=1):\n", corr_with_target.head(10))
print("Top 10 negatively correlated (i.e., with malignant y=0):\n", corr_with_target.tail(10))