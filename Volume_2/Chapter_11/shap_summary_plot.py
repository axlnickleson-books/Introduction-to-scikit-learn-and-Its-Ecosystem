# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 01:24:49 2025

@author: Admin
"""

import shap
import xgboost
import numpy as np
import matplotlib.pyplot as plt
X, y = shap.datasets.adult()
model = xgboost.XGBClassifier().fit(X, y)
explainer = shap.Explainer(model)
shap_values = explainer(X)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()