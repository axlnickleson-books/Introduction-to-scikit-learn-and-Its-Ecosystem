from sklearn.metrics import mean_squared_log_error
import numpy as np
# True targets and predictions
y_true = [3, 5, 2.5 , 7]
y_pred = [2.5 , 5, 4, 8]
# 1) Using scikit - learn 's built -in function
msle_sklearn = mean_squared_log_error( y_true , y_pred )

# 2) Manual computation for verification
log_true = np.log1p( y_true ) # log (1 + y_true )
log_pred = np.log1p( y_pred ) # log (1 + y_pred )
squared_log_errors = ( log_true - log_pred ) **2
msle_manual = squared_log_errors.mean ()
print (f"Log - transformed true : { log_true }")
print (f"Log - transformed pred : { log_pred }")
print (f" Squared log errors : { squared_log_errors }")
print (f" MSLE ( sklearn ) = { msle_sklearn }")
print (f" MSLE ( manual ) = { msle_manual }")