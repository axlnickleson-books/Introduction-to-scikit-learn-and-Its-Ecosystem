from sklearn.metrics import roc_auc_score 
# True labels and predicted labels 
y_real = [0, 1, 1, 0, 1, 1, 0]
y_predict = [0, 1, 1, 0, 1, 1, 1]
# Compute ROC AUC using sklearn 
AUC = roc_auc_score(y_real, y_predict)
print("AUC (sklearn) = ", AUC)

#Manual implementation/calculation with NumPy 
import numpy as np
# True labels and predicted labels
y_true = np.array ([0 , 1, 1, 0, 1, 1, 0])
y_pred = np.array ([0 , 1, 1, 0, 1, 1, 1])
# Step 1: Sort by predicted values ( descending )
order = np. argsort (- y_pred )
y_true_sorted = y_true [ order ]
# Step 2: Calculate cumulative TPR and FPR
P = np.sum( y_true == 1)
N = np.sum( y_true == 0)
TP = np.cumsum( y_true_sorted == 1)
FP = np.cumsum( y_true_sorted == 0)
TPR = TP / P
FPR = FP / N
# Step 3: Add starting point (0 ,0)
TPR = np.insert(TPR , 0, 0)
FPR = np.insert(FPR , 0, 0)
# Step 4: Calculate area under curve ( trapezoidal rule )
auc_manual = np.trapz(TPR , FPR )
print(" TPR =", TPR )
print(" FPR =", FPR )
print(" AUC ( manual ) =", auc_manual )