#Part 2
#One hot encoding the features and lable
from keras.utils import to_categorical
from sklearn import preprocessing
num_classes =2 # Two class classification
enc = preprocessing.OneHotEncoder()
enc.fit(x_train)
x_train_onehot = enc.transform(x_train).toarray()
enc.fit(x_test_original)
x_test_onehot= enc.transform(x_test_original).toarray()
# One hot label
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_test_onehot  =  keras.utils.to_categorical(y_test_original, num_classes)