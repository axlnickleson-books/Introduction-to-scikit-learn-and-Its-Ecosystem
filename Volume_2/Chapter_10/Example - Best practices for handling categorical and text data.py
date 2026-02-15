# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 03:38:47 2025

@author: Admin
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Small example dataset
corpus = [
    "scikit-learn provides easy-to-use ML tools",
    "machine learning with python is powerful",
    "data preprocessing is essential",
    "python and scikit-learn are great for ML",
    "text data requires proper preprocessing",
    "machine learning models benefit from tfidf"
]

# Labels (binary for demonstration)
y = [1, 0, 0, 1, 0, 0]

# Split the corpus
X_train, X_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.33, random_state=42
)

# Build the pipeline
text_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1,2))),
    ('reducer', TruncatedSVD(n_components=5, random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit the pipeline
text_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = text_pipeline.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Show predicted labels
print("Predicted labels:", y_pred)

# Show transformed TF-IDF → SVD representation
X_test_transformed = text_pipeline.named_steps["reducer"].transform(
    text_pipeline.named_steps["vectorizer"].transform(X_test)
)

print("\nSVD-transformed feature matrix (test samples):")
print(X_test_transformed)

