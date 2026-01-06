from sklearn . metrics import accuracy_score
y_real = [0 ,1 ,1 ,0 ,1 ,1 ,0]
y_predict = [0 ,1 ,1 ,0 ,1 ,1 ,1]
ACC = accuracy_score ( y_real , y_predict )
print (" ACC = {}". format ( ACC ))