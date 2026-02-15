# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:55:41 2025

@author: Admin
"""

from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time	
X, y = make_classification(
	n_samples=50000,
	n_features=20,
	random_state=42
	)
	
X_train, X_test, y_train, y_test = train_test_split(
	X, y, random_state=42
	)
start = time.time()
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train, y_train)
print("Accuracy (XGBoost):", accuracy_score(y_test, xgb.predict(X_test)))
print("Training Time (s):", time.time() - start)