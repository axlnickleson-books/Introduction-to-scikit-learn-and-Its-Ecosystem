# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:18:29 2025

@author: Admin
"""

import re
import numpy as np
import pandas as pd
	
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
	
#	-------------------------------------------
#	0) Start from the raw Titanic DataFrame df
#	(assumes columns: Survived, Pclass, Name, Sex, Age, SibSp, Parch,
#	Ticket, Fare, Cabin, Embarked)
#	-------------------------------------------
#	y: target; X: raw features (no manual preprocessing here)
	
y = df["Survived"].copy()
X = df.drop(columns=["Survived"])
	
# 	-------------------------------------------
# 	1) Custom feature builder (inside the pipeline)
# 	- CabinKnown (binary)
# 	- FamilySize, FarePerPerson (numerical)
# 	- Title extraction + consolidation (categorical)
# 	- Drop raw Cabin/Ticket/Name after extracting signals
# 	-------------------------------------------
	
class FeatureBuilder(BaseEstimator, TransformerMixin):
    def init(self, rare_threshold=10):
        self.rare_threshold = rare_threshold
        self.title_map_ = {
		"Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
		"Lady":"Royalty","Countess":"Royalty","Sir":"Royalty",
		"Don":"Royalty","Dona":"Royalty","Jonkheer":"Royalty",
		"Capt":"Officer","Col":"Officer","Major":"Officer",
		"Dr":"Officer","Rev":"Officer"
	    }  
    self.rare_titles_ = None
    def fit(self, X, y=None):
	  Xc = X.copy()
	# Extract Title
	titles = Xc["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
	titles = titles.replace(self.title_map_)
	vc = titles.value_counts()
	self.rare_titles_ = set(vc[vc < self.rare_threshold].index)
	return self
	
	def transform(self, X):
	Xc = X.copy()
	
	# Binary cabin indicator
	Xc["CabinKnown"] = Xc["Cabin"].notna().astype(int)
	
	# Family features
	Xc["FamilySize"] = Xc["SibSp"] + Xc["Parch"] + 1
	Xc["FarePerPerson"] = Xc["Fare"] / Xc["FamilySize"]
	
	# Title from Name (then consolidate and group rares)
	titles = Xc["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
	titles = titles.replace(self.title_map_)
	titles = titles.where(~titles.isin(self.rare_titles_), other="Rare")
	Xc["Title"] = titles
	
	# Drop raw text-ish/ID columns we no longer need
	Xc = Xc.drop(columns=["Cabin","Ticket","Name"], errors="ignore")
	return Xc
	
# 	-------------------------------------------
# 	2) Define column groups as they exist AFTER FeatureBuilder
# 	-------------------------------------------
	
	numeric_features = [
	"Pclass","Age","SibSp","Parch","Fare",
	"FamilySize","FarePerPerson","CabinKnown"
	]
	categorical_features = ["Sex","Embarked","Title"]
	
# 	-------------------------------------------
# 	3) Preprocessing for each group
# 	- Numeric: median impute + scale (beneficial for LR)
# 	- Categorical: most-frequent impute + one-hot (drop first to avoid collinearity)
# 	-------------------------------------------
	
	numeric_pipe = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="median")),
	("scaler", StandardScaler())
	])
	
	categorical_pipe = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="most_frequent")),
	("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
	])
	
	preprocessor = ColumnTransformer(
	transformers=[
	("num", numeric_pipe, numeric_features),
	("cat", categorical_pipe, categorical_features)
	],
	remainder="drop"
	)
	
# 	-------------------------------------------
# 	4) Full pipeline: FeatureBuilder -> ColumnTransformer -> LogisticRegression
# 	-------------------------------------------
	
	clf = Pipeline(steps=[
	("features", FeatureBuilder(rare_threshold=10)),
	("preprocess", preprocessor),
	("model", LogisticRegression(max_iter=1000))
	])
	
# 	-------------------------------------------
# 	5) Train/test split + fit + evaluate
# 	-------------------------------------------
	
	X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, stratify=y, random_state=42
	)
	
	clf.fit(X_train, y_train)
	
	print("Accuracy:", clf.score(X_test, y_test))
	
	y_pred = clf.predict(X_test)
	y_proba = clf.predict_proba(X_test)[:, 1]
	
	print("\nClassification report:")
	print(classification_report(y_test, y_pred, digits=3))
	
	print("ROC-AUC:", roc_auc_score(y_test, y_proba))