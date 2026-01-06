from sklearn . metrics import recall_score
# True labels and predicted labels
y_real = [0, 1, 1, 0, 1, 1, 0]
y_predict = [0, 1, 1, 0, 1, 1, 1]
# Calculate recall
Rec = recall_score ( y_real , y_predict )
print (" Rec = {}". format ( Rec ))