from sklearn . metrics import mean_squared_error
import numpy as np
# True targets and predictions
y_true = [3, -0.5 , 2, 7]
y_pred = [2.5 , 0.0 , 2, 8]
# 1) Compute MSE using scikit - learn
mse = mean_squared_error ( y_true , y_pred )
# 2) Take square root to obtain RMSE
rmse = np.sqrt( mse )
# 3) Manual computation for verification
squared_errors = (np. array( y_true ) - np.array( y_pred ))**2
rmse_manual = np.sqrt(squared_errors.mean())
print (f" Squared errors : { squared_errors.tolist()}")
print (f" RMSE ( from MSE ) = { rmse }")
print (f" RMSE ( manual ) = { rmse_manual }")