# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 07:05:29 2017

@author: Eric Vos
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#Load MNIST Dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
#Cook Model
clf = MLPClassifier(hidden_layer_sizes = [20, 10], alpha = 5.0,random_state = 0, solver='lbfgs').fit(X_train, y_train)
#Taste Model
y_pred = clf.predict(X_test)
print('MNIST')
print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test, y_pred)))