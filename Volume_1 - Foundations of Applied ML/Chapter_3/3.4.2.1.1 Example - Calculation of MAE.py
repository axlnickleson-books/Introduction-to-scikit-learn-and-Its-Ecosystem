from sklearn.metrics import mean_absolute_error
import numpy as np
# True targets and predictions
y_true = [3, -0.5 , 2, 7]
y_pred = [2.5 , 0.0 , 2, 8]
# 1) Using scikit - learn 's built -in function
mae_sklearn = mean_absolute_error( y_true , y_pred )
# 2) Manual computation with NumPy for verification
abs_errors = np.abs (np.array( y_true ) - np.array( y_pred ))
mae_manual = abs_errors.mean ()
print (f" Absolute errors : { abs_errors.tolist()}")
print (f" MAE ( sklearn ) = { mae_sklearn }")