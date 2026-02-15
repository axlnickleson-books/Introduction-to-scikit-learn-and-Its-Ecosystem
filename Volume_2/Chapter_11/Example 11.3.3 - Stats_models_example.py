# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 01:14:36 2025

@author: Admin
"""

import statsmodels.api as sm
import numpy as np

X = np.random.rand(100, 3)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())