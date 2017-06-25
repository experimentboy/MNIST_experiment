# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 07:05:29 2017

@author: Eric Vos
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#Load MNIST Dataset
data = pd.read_csv('train.csv')
y = data['label']
del data['label']
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.75)
#Cook Model
clf = MLPClassifier(hidden_layer_sizes = [200, 200], alpha = 5.0,random_state = 0, solver='lbfgs').fit(X_train, y_train)
#Taste Model
y_pred = clf.predict(X_test)
#Full testing
X_test2 = pd.read_csv('test.csv')
y_test2 = clf.predict(X_test2)
#X_test3 = pd.read_csv('sample_submission.csv')
#X_test3['Label'] = X_test2['Label']
print('MNIST')
print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test, y_pred)))
#X_test3.drop(X_test3.columns[[0]], axis=1)
#X_test3.to_csv('MNIST_Predict_NN11.csv',sep=',', encoding='utf-8')
np.savetxt('submission2.csv', np.c_[range(1,len(X_test2)+1),y_test2], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')