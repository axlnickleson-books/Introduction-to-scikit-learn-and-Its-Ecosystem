# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 03:23:12 2025

@author: Admin
"""

# ---------------------------------------------------------
# Example: Training and Evaluating a Text Classifier with NB
# ---------------------------------------------------------

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Load the 20 Newsgroups dataset
# -----------------------------
train = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)
test = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

# -----------------------------
# Vectorize text using TF–IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train = vectorizer.fit_transform(train.data)
X_test  = vectorizer.transform(test.data)

# -----------------------------
# Train Naive Bayes classifier
# -----------------------------
clf = MultinomialNB()
clf.fit(X_train, train.target)

# -----------------------------
# Evaluate performance
# -----------------------------
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(test.target, y_pred))
print("\nClassification Report:")
print(classification_report(test.target, y_pred))

# Optional: Confusion matrix for deeper insight
cm = confusion_matrix(test.target, y_pred)
print("Confusion Matrix:\n", cm)
