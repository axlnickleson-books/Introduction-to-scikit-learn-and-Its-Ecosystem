###################################################
# Chapter 1 - What is scikit-learn? 
# Quick Start in 10 Lines 
##################################################
# Step 1 - Import Libraries 
##################################################
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
##################################################
# Step 2 - Load the Breast Cancer Dataset
##################################################
X, y = load_breast_cancer(return_X_y=True)
print("X = {}".format(X))
print("y = {}".format(y))
print("X shape = {}".format(X.shape))
print("y shape = {}".format(y.shape))
##################################################
# Step 3 - Split the Dataset
##################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("X_train shape = {}".format(X_train.shape))
print("X_test shape = {}".format(X_test.shape))
print("y_train shape = {}".format(y_train.shape))
print("y_test shape = {}".format(y_test.shape))
##################################################
# Step 4 - Define and Train the Random Forest Classifer
##################################################
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
##################################################
# Step 5 - Evaluate the Model and Show the Results
##################################################
y_predict = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predict))