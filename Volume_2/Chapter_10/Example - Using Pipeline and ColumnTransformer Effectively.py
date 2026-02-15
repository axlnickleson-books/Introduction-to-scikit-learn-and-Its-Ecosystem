# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 03:45:49 2025

@author: Admin
"""


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import pandas as pd
import numpy as np

# Expanded example dataset (balanced, avoids warnings)
df = pd.DataFrame({
    'age':    [25, 32, 47, 51, 29, 41, 53, 38, 27, 49, 45, 60],
    'income': [50000, 64000, 120000, 98000, 52000, 110000, 75000, 68000, 58000, 130000, 90000, 102000],
    'gender': ['M','F','F','M','F','M','M','F','F','M','F','M'],
    'purchased': [0,1,1,0,0,1,0,1,0,1,1,0]
})

X = df[['age', 'income', 'gender']]
y = df['purchased']

# Define transformers
num_features = ['age', 'income']
cat_features = ['gender']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Split so both classes appear in train AND test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=42
)

clf.fit(X_train, y_train)

print("Model accuracy:", clf.score(X_test, y_test))

# ---------------------------------------------------------
# Additional outputs
# ---------------------------------------------------------

# 1. Transformed matrix
print("\n=== Transformed Feature Matrix (train) ===")
X_train_transformed = clf.named_steps['preprocessor'].transform(X_train)
print(X_train_transformed)

# 2. Feature names
print("\n=== Transformed Feature Names ===")
num_names = num_features
cat_names = clf.named_steps['preprocessor'] \
               .named_transformers_['cat'] \
               .named_steps['encoder'] \
               .get_feature_names_out(cat_features)
print(list(num_names) + list(cat_names))

# 3. Predictions
y_pred = clf.predict(X_test)
print("\n=== Predictions ===")
print(y_pred)

# 4. Probabilities
print("\n=== Prediction Probabilities ===")
print(clf.predict_proba(X_test))

# 5. Confusion matrix
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 6. Classification report (NO WARNINGS NOW)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
