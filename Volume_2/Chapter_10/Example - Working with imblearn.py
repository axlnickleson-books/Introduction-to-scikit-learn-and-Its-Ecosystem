# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 00:32:52 2025

@author: Admin
"""

# === Import libraries ===
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import Counter

# === 1. Load a real dataset (Breast Cancer) ===
data = load_breast_cancer()
X = data.data
y = data.target

print("Original class distribution:", Counter(y))
# Note: 0 = malignant, 1 = benign

# === 2. Create train/test split BEFORE oversampling ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train distribution before SMOTE:", Counter(y_train))

# === 3. Standardize features (good practice for distance-based methods) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. Apply SMOTE only on the training set ===
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

print("Train distribution after SMOTE:", Counter(y_res))

# === 5. Train a classifier ===
clf = RandomForestClassifier(random_state=42)
clf.fit(X_res, y_res)

# === 6. Evaluate on the ORIGINAL (untouched) test set ===
y_pred = clf.predict(X_test_scaled)
print("\n=== Classification Report (Evaluation on Original Test Set) ===")
print(classification_report(y_test, y_pred))
