# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:14:41 2025

@author: Admin
"""

from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.datasets import load_breast_cancer
	
# Train a simple model
X, y = load_breast_cancer(return_X_y=True)
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)
	
# Save the trained model
dump(clf, "rf_model.joblib")

# Later, load it back
rf_loaded = load("rf_model.joblib")
print(rf_loaded.predict(X[:5]))