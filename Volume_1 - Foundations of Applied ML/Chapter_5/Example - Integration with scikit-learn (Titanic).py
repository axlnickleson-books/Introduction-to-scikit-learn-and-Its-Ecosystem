# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 00:19:31 2025

@author: Admin
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             classification_report)

data = pd.read_csv("Titanic_ready.csv")
print("Dataset preview:")
print(data.head())
X = data.copy()
y = X.pop('Survived')
#Split DataFrame into train/test
	
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
	)
	
#Train classifier
	
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
	
y_pred = clf.predict(X_test)
# Step 7 — Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))