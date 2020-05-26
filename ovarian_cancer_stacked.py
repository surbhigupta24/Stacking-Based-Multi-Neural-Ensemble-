# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:04:48 2020

@author: SURBHI
"""


""" ***************************************************************************
# * File Description:                                                         *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Scale Data                                                             *
# * 3. Create train and test set                                              *
# * 4. MLP_1                                                                  *
# * 5. MLP_2                                                                  *
# * 6. MLP_3                                                                  *
# * 7. Stacking                                                               *
# * AUTHORS(S): SURBHI <sur7312@gmail.com>                                    *
# * --------------------------------------------------------------------------*
# * ************************************************************************"""


###############################################################################
#                     1. Importing Libraries                                  #                                      
###############################################################################

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier 
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
#from imblearn.ensemble import EasyEnsemble, BalanceCascade
#import smote_variants as sv
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import warnings
warnings.simplefilter('ignore')


data = pd.read_excel('ovarian_code_.xlsx')
data.isna().sum()

data = data.drop(columns = ['HEIGHT', 'CASTE','RELIGION ','PLACE OF ORIGIN' ])
data.isna().sum()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1]

data1 = data.iloc[:, :-1]

from missingpy import KNNImputer 
cols = list(data1)

data1 = pd.DataFrame(KNNImputer().fit_transform(data1))
data1.columns = cols

data1.isna().sum()        

data2 = pd.concat([data1, y], axis = 1) 
data2 = data2.dropna()

X = data2.iloc[:, :-1].values
y = data2.iloc[:, -1].values

y = [0 if i == 1 else 1 for i in y]
    

###############################################################################
#                        2. Scale Data                                         #                                      
###############################################################################

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

SENN = SMOTEENN()
ennx, enny = SENN.fit_sample(X, y)

###############################################################################
#                        3. Create train and test set                         #
###############################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)


###################################################################################
trainX = X_train
trainy = y_train
testX = X_test
testy = y_test

###############################################################################
#                                  4. MLP_1                                   #
###############################################################################
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 50, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 8))

model.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

import os
folder = 'models'
path1 = os.getcwd()
if folder not in os.listdir(path1):
    os.makedirs(folder)

filename = 'models/model_' + str(0) + '.h5'
model.save(filename)
print('>Saved %s' % filename)
    

###############################################################################
#                                  5. MLP_2                                    #
###############################################################################

model1 = Sequential()
model1.add(Dense(units = 150, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim =8))

model1.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

model1.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

model1.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))


model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
  
history = model1.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
_, train_acc = model1.evaluate(trainX, trainy, verbose=0)
_, test_acc = model1.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

filename = 'models/model_' + str(1) + '.h5'
model1.save(filename)
print('>Saved %s' % filename)

###############################################################################
#                                  6. MLP_3                                    #
###############################################################################

model2 = Sequential()
model2.add(Dense(units = 200, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 8))

model2.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

model2.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

model2.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))
model2.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

model2.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
model2.fit(trainX, trainy, epochs=500, verbose=0)

history = model2.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
_, train_acc = model2.evaluate(trainX, trainy, verbose=0)
_, test_acc = model2.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

filename = 'models/model_' + str(2) + '.h5'
model2.save(filename)
print('>Saved %s' % filename)

n_model = 3
from keras.models import load_model
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		filename = 'models/model_' + str(i) + '.h5'
		classifier = load_model(filename)
		all_models.append(classifier)
		print('>loaded %s' % filename)
	return all_models


n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))


for classif in members:
	_, acc = classif.evaluate(testX, testy, verbose=0)
	print('Model Accuracy: %.3f' % acc)


###############################################################################
#                              7. Stacking                                    #
###############################################################################
    
from numpy import dstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
import statistics

def stacked_dataset(members, inputX):
	stackX = None
	for classifier in members:
		yhat = classifier.predict(inputX, verbose=0)
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

def fit_stacked_model(members, inputX, inputy):
	stackedX = stacked_dataset(members, inputX)
	model = GradientBoostingClassifier()
	model.fit(stackedX, inputy)
	return model
# model = fit_stacked_model(members, trainX, trainy)

def stacked_prediction(members, model, inputX):
	stackedX = stacked_dataset(members, inputX)
	yhat = model.predict(stackedX)
	return yhat
# yhat = stacked_prediction(members, model, testX)

accuracy = []
f1_scores = []
mcc_score = []
auc_score = []
sensitivity_score = []
specificity_score = []

cv = KFold(n_splits = 10, random_state = 42, shuffle = True)
for train_index, test_index in cv.split(ennx):
    X_train1, X_test1, y_train1, y_test1 = ennx[train_index], ennx[test_index], enny[train_index], enny[test_index]

    model = fit_stacked_model(members, X_train1, y_train1)
    yhat = stacked_prediction(members, model, X_test1)

    acc = accuracy_score(y_test1, yhat)
    print(acc)
    f1 = f1_score(y_test1, yhat)
    mcc = matthews_corrcoef(y_test1, yhat)
    auc = roc_auc_score(y_test1, yhat)
    cm = confusion_matrix(y_test1, yhat)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    
    accuracy.append(acc)
    f1_scores.append(f1)
    mcc_score.append(mcc)
    auc_score.append(auc)
    sensitivity_score.append(sensitivity)
    specificity_score.append(specificity)
    
print("Accuracy: " + statistics.mean(accuracy).__str__())
print("AUC: " + statistics.mean(f1_scores).__str__())
print("f1: " + statistics.mean(mcc_score).__str__())
print("MCCC: " + statistics.mean(auc_score).__str__())
print("Sensitivity: " + statistics.mean(sensitivity_score).__str__())
print("Specficity: " + statistics.mean(specificity_score).__str__())


