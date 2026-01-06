from sklearn . metrics import precision_score
# True labels and predicted labels
y_real = [0, 1, 1, 0, 1, 1, 0]
y_predict = [0, 1, 1, 0, 1, 1, 1]
# Calculate precision
Prec = precision_score ( y_real , y_predict )
print (" Prec = {}". format ( Prec ))