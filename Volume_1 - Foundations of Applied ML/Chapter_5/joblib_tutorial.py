# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:14:37 2025

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

# Save with compression
dump(clf, "rf_model_compressed.joblib", compress=3)
	
# Alternative compression formats
# dump(clf, "rf_model_lz4.joblib", compress=("lz4", 3))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
	
pipe = Pipeline([
("scaler", StandardScaler()),
("model", LogisticRegression())
])
	
pipe.fit(X, y)
	
# Save entire pipeline
dump(pipe, "logreg_pipeline.joblib")

# Load and predict
loaded_pipe = load("logreg_pipeline.joblib")
print(loaded_pipe.predict(X[:5]))