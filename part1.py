# Part 1. Use pandas to read data and extract features and label. Then over_sampling and split to train and test.

#Import pandas and read data
import pandas as pd 
data = pd.read_csv('dataPlus.csv')
feature = data1.drop(['diag_ebd'],axis =1)
label = data1['diag_ebd']

#Import and use the SMOTE to over sampling the data ratio to 1:1
from imblearn.over_sampling import SMOTE
import numpy as np
sm = SMOTE(random_state =12, ratio =1.0)
feature_over_sampling, label_over_sampling = sm.fit_sample(feature,label)
#Round up to interger
feature_over_sampling= np.round(feature_over_sampling)

#Split dataset
from sklearn.model_selection import train_test_split
#Split original data to sample 1 and sample 2 mentioned randomgly in the article
x_train_original, x_test_original, y_train_original, y_test_original= train_test_split(feature_over_sampling[0:2459], feature_over_sampling[0:2459], test_size =0.25, random_state =12)
#Split oversampling data randomly to get sample 3
x_train_oversampling, x_test_oversampling, y_train_oversampling, y_test_oversampling= train_test_split(feature_over_sampling[2459:], feature_over_sampling[2459:], test_size =0.25, random_state =12)
# Concatenate sample 1 and sample 3 
x_train = np.concatenate((x_train_original, x_train_oversampling))
y_train = np.concatenate((y_train_original, y_train_oversampling))
