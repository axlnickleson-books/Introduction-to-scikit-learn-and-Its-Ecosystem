# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 04:09:20 2025

@author: Admin
"""

from skopt import BayesSearchCV
from skopt.plots import plot_convergence
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# 1. Load dataset and split properly
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Define search space
search_space = {
    'C': (1e-3, 1e3, 'log-uniform'),
    'gamma': (1e-4, 1e0, 'log-uniform'),
    'kernel': ['linear', 'rbf']
}

# 3. Run Bayesian optimization
bayes_search = BayesSearchCV(
    SVC(),
    search_space,
    n_iter=25,
    cv=5,
    random_state=42,
    n_jobs=-1
)
bayes_search.fit(X_train, y_train)

# 4. Show best parameters and score
print("Best Parameters:", bayes_search.best_params_)
print("Best CV Score:", bayes_search.best_score_)

# 5. Evaluate best model
y_pred = bayes_search.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# 6. Plot convergence curve
plt.figure(figsize=(12,8))
plot_convergence(bayes_search.optimizer_results_[0])
plt.savefig("bayes_convergence_plot.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()
