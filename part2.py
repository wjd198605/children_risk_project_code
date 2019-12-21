#Part 2 One-hot encoding the sample 1, 2 and 3 and build all three models

#One-hot encoding features
from keras.utils import to_categorical
from sklearn import preprocessing
num_classes =2 # Two class classification
enc = preprocessing.OneHotEncoder()
enc.fit(x_train)
x_train_onehot = enc.transform(x_train).toarray()
enc.fit(x_test_original)
x_test_onehot= enc.transform(x_test_original).toarray()
#One-hot encoding label
y_train_onehot = keras.utils.to_categorical(y_train_original, num_classes)
y_test_onehot  =  keras.utils.to_categorical(y_test_original, num_classes)

#Import and build models

#Import Naïve Bayes and Logistic Regression models from the SKlearn module
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#Import Deep Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import RMSprop
#Build Naïve Bayes model 
gnb = GaussianNB()
#Build Logistic Regression model
lr = LogisticRegression()
#Buld Deep Neural Network
dnn = Sequential()
dnn.add(Dense(32, activation='relu', input_dim=123)) # input_dim is the input shape corresponding to the dataset
dnn.add(Dense(16,activation='relu'))
dnn.add(Dense(8,activation='relu'))
dnn.add(Dense(num_classes, activation='sigmoid'))
dnn.compile(loss='categorical_crossentropy',
                 optimizer=RMSprop(),
               metrics=['accuracy'])
#Trian all three models with sample 1 and sample 3
#Naïve Bayes and Logistic Regression
gnb.fit(x_train_onehot,y_train_original)
lr.fit(x_train_onehot,y_train_original)
#Deep Neural Network
dnn.fit(x_train_onehot,y_train_onehot,
                    batch_size=4,
                    epochs=30,
                    verbose=1,                     
                    )

