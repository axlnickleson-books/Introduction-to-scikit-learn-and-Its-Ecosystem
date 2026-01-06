from sklearn . metrics import f1_score
import numpy as np
# True labels and predicted labels
y_true = np. array ([0 , 1, 1, 0, 1, 1, 0])
y_pred = np. array ([0 , 1, 1, 0, 1, 1, 1])
# 1) F1 - score using scikit - learn
f1_sklearn = f1_score ( y_true , y_pred )
# 2) Manual calculation with NumPy
TP = np. sum (( y_true == 1) & ( y_pred == 1))
FP = np. sum (( y_true == 0) & ( y_pred == 1))
FN = np. sum (( y_true == 1) & ( y_pred == 0))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_manual = 2 * ( precision * recall ) / ( precision + recall )
print(f"TP ={ TP}, FP ={ FP}, FN ={ FN}")
print(f" Precision ={ precision }, Recall ={ recall }")
print(f"F1 ( sklearn ) = { f1_sklearn }")
print(f"F1 (manual) = {f1_manual}")