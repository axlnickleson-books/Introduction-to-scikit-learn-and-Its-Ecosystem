# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 03:28:41 2025

@author: Admin
"""

# from tpot import TPOTClassifier
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# 	
# X, y = load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# 	
# tpot = TPOTClassifier(generations=3, population_size=10)
# tpot.fit(X_train, y_train)
# y_predict = tpot.predict(X_test)
# ACC = accuracy_score(y_test, y_predict)
# print("AutoML Accuracy:", ACC)


from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Run TPOT
tpot = TPOTClassifier(generations=3, population_size=10, verbose=1, random_state=42)
tpot.fit(X_train, y_train)

# Extract pipeline fitness values (approximation)
scores = [pipeline['internal_cv_score'] for pipeline in tpot.evaluated_individuals_.values()]
scores.sort(reverse=True)

# Simulate “best per generation” for visualization
generations = list(range(1, len(scores[:4]) + 1))
best_scores = scores[:4]
avg_scores = [sum(scores[:i]) / i for i in generations]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(generations, best_scores, marker='o', label='Best Pipeline Score')
plt.plot(generations, avg_scores, marker='s', linestyle='--', label='Average Pipeline Score')
plt.title("Evolution of Pipelines during TPOT Optimization")
plt.xlabel("Generation")
plt.ylabel("Cross-Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
