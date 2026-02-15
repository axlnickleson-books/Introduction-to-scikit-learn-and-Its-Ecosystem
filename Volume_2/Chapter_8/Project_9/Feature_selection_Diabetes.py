# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 12:58:33 2025

@author: Admin
"""
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = load_diabetes()
X, y = data.data, data.target
	
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
	
print("Shape:", X.shape, y.shape)
print("First 5 rows:\n", df.head())

selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)
	
selected_features = np.array(data.feature_names)[selector.get_support()]
print("Top 5 features by univariate test:", selected_features)

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)
	
coef = pd.Series(lasso.coef_, index=data.feature_names)
print("Lasso coefficients:\n", coef)
	
# Visualize
coef.plot(kind="barh", figsize=(8, 6), 
          color="black",  # all bars in black
          title="Feature Importance via Lasso",
          zorder=3)
plt.grid(True, axis="x", color="gray", linestyle="--", linewidth=0.7, zorder=0)
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.show()
# plt.savefig("FeatureImportanceViaLasso.png",
#             dpi = 300,
#             bbox_inches="tight")
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)
	
importances = pd.Series(rf.feature_importances_, index=data.feature_names)
print("Random Forest feature importances:\n", importances.sort_values(ascending=False))
	
importances.sort_values().plot(kind="barh", 
                               color="black",
                               figsize=(8,5), 
                               title="Random Forest Feature Importance",
                               zorder=3)
plt.grid(True, axis="x", color="gray", linestyle="--", linewidth=0.7, zorder=0)
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
# plt.show()
plt.savefig("FeatureImportanceViaRFR.png",
            dpi = 300,
            bbox_inches="tight")