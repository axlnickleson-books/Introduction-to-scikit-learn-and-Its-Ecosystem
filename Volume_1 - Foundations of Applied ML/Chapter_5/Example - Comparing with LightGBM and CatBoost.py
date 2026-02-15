# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 14:56:19 2025

@author: Admin
"""

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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
lgb = LGBMClassifier(verbose=0)
cat = CatBoostClassifier(verbose=0)
	
for model in [lgb, cat]:
	start = time.time()
	model.fit(X_train, y_train)
	acc = accuracy_score(y_test, model.predict(X_test))
	print(f"{model.__class__.__name__} - Accuracy: {acc:.3f}, Time: {time.time() - start:.2f}s")