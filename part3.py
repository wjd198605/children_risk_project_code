#Examine model performance 
from sklearn.metrics import confusion_matrix

#Examine model with sample 2

#Logistic Regression
y_predict_lr = lr.predict(x_test_onehot)
#Compare the predicted label with the original label
cnf_matrix_lr = confusion_matrix(y_test_original,y_predict_lr)

#Naïve Bayes
y_predict_gnb = gnb.predict(x_test_onehot)
#Compare the predicted label with the original label
cnf_matrix_gnb= confusion_matrix(y_test_original,y_predict_gnb)

#Deep Neural Network
y_predict_dnn= dnn.predict_classes(x_test_onehot) #Predict the label for the text 
#y_train_predict = model.predict(x_train)
cnf_matrix_dnn= confusion_matrix(y_test_original, y_predict_dnn)