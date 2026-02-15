# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 04:00:39 2025

@author: Admin
"""

import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

# --------------------------------------------------------
# 1. Set all relevant seeds for reproducibility
# --------------------------------------------------------
np.random.seed(42)
random.seed(42)

# --------------------------------------------------------
# 2. Load dataset
# --------------------------------------------------------
X, y = load_wine(return_X_y=True)

# --------------------------------------------------------
# 3. Reproducible train/test split
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

# --------------------------------------------------------
# 4. Train a reproducible Random Forest
# --------------------------------------------------------
model_fixed = RandomForestClassifier(random_state=42)
model_fixed.fit(X_train, y_train)

y_pred_fixed = model_fixed.predict(X_test)
acc_fixed = accuracy_score(y_test, y_pred_fixed)

print("Accuracy with fixed random_state=42:", acc_fixed)

# --------------------------------------------------------
# 5. Train a Random Forest without fixed random state
#    (results will vary run-to-run)
# --------------------------------------------------------
model_unfixed = RandomForestClassifier()   # random_state=None
model_unfixed.fit(X_train, y_train)

y_pred_unfixed = model_unfixed.predict(X_test)
acc_unfixed = accuracy_score(y_test, y_pred_unfixed)

print("Accuracy WITHOUT random_state:", acc_unfixed)

# --------------------------------------------------------
# 6. Demonstrate that fixed seeds produce identical predictions
# --------------------------------------------------------
model_fixed2 = RandomForestClassifier(random_state=42)
model_fixed2.fit(X_train, y_train)

y_pred_fixed2 = model_fixed2.predict(X_test)

print("\nPredictions identical?",
      np.array_equal(y_pred_fixed, y_pred_fixed2))