from sklearn.metrics import mean_squared_error 
import numpy as np 
# True targets and predictions 
y_true = [3, -0.5 , 2, 7]
y_pred = [2.5 , 0.0 , 2, 8]
# 1) Using scikit learn's buit in function 
mse_sklearn = mean_squared_error(y_true, y_pred)

# 2) Manual computation with NumPy for verification 
squared_errors = (np.array(y_true) - np.array(y_pred))**2 
mse_manual = squared_errors.mean()
print(f" Squared errors : { squared_errors.tolist()}")
print(f" MSE ( sklearn ) = { mse_sklearn }")
print(f" MSE ( manual ) = { mse_manual }")