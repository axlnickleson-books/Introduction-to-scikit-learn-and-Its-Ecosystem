# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 03:27:52 2025

@author: Admin
"""


from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
	
X_full, y = fetch_openml("titanic", version=1, as_frame=True, parser="pandas").data, fetch_openml("titanic", version=1, as_frame=True, parser="pandas").target
X = X_full[["pclass", "sex", "age", "fare"]]
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X[["age", "fare"]])
	

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[["sex"]])
	
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
	
X_final = np.hstack((X_encoded, X_scaled))
print("Final transformed shape:", X_final.shape)