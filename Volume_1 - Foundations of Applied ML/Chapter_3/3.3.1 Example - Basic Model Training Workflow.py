########################################################
# Step 1 - Import Required Libraries
########################################################
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
########################################################
# Step 2 - Load the Dataset
########################################################	
X, y = load_iris(return_X_y=True)
########################################################
# Step 3 - Split the Data into Training and Test Sets
########################################################	
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
	)
########################################################
# Step 4 - Initialize the Model 
########################################################
model = RandomForestClassifier()
########################################################
# Step 5 - Selecting a Model 
########################################################
model.fit(X_train, y_train)