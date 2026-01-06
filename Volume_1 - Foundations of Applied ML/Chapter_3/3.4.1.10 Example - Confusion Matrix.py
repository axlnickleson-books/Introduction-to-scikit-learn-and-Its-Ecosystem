from sklearn.metrics import confusion_matrix 
y_real = [0,1,1,0,1,1,0]
y_predict = [0, 1, 1, 0, 1, 1, 1] 
CM = confusion_matrix(y_real, y_predict)
print("CM = {}".format(CM))