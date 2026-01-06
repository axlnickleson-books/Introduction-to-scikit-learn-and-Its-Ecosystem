from sklearn.metrics import matthews_corrcoef
import numpy as np
# True labels and predicted labels
y_true = np.array([0 , 1, 1, 0, 1, 1, 0])
y_pred = np.array([0 , 1, 1, 0, 1, 1, 1])
# 1) MCC using scikit - learn
mcc_sklearn = matthews_corrcoef ( y_true , y_pred )
# 2) Manual MCC calculation with NumPy
TP = np.sum(( y_true == 1) & ( y_pred == 1))
TN = np.sum(( y_true == 0) & ( y_pred == 0))
FP = np.sum(( y_true == 0) & ( y_pred == 1))
FN = np.sum(( y_true == 1) & ( y_pred == 0))
numerator = (TP * TN) - (FP * FN)
denominator = np.sqrt (( TP + FP) * (TP + FN) * (TN + FP) * (TN +FN))
mcc_manual = numerator / denominator
print(f"TP ={ TP}, TN ={ TN}, FP ={ FP}, FN ={ FN}")
print(f" MCC ( sklearn ) = { mcc_sklearn }")
print(f" MCC ( manual ) = { mcc_manual }")