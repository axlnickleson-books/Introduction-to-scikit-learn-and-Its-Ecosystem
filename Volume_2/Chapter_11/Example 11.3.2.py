# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 00:36:58 2025

@author: Admin
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
	
def create_model():
	model = Sequential([
	Dense(64, activation='relu', input_dim=20),
	Dense(32, activation='relu'),
	Dense(1, activation='sigmoid')
	])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model
	
X = np.random.rand(500, 20)
y = np.random.randint(0, 2, 500)
	
keras_estimator = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)
scores = cross_val_score(keras_estimator, X, y, cv=3)
print("Average CV Accuracy:", scores.mean())