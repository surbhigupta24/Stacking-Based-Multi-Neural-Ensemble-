# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:27:55 2019
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
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from imblearn.combine import SMOTEENN

data = pd.read_excel('lung_jk.xlsx')
data.isna().sum()
data = data.drop(columns = ['COUGH (Days)', 'HOARNESS OF VOICE', 'BREATHLESSNESS/ SHORTNESS OF BREATH/DYSPNEA',
                            'FEVER', 'WEAKNESS& WEIGHTLOSS'])
data1 = data

from missingpy import KNNImputer 
cols = list(data1)

data1 = pd.DataFrame(KNNImputer().fit_transform(data1))
data1.columns = cols
data1.isna().sum()
data1.info()

data1['STAGE'].value_counts()

for i in data1.index:
    if data1['STAGE'][i] == 1.0:
        data1['STAGE'][i] = np.nan
    else:
        pass
  
data1 = data1.dropna()

for i in data1.index:
    if data1['STAGE'][i] == 2.0:
        data1['STAGE'][i] = 0
    elif data1['STAGE'][i] == 3.0:
        data1['STAGE'][i] = 1
    elif data1['STAGE'][i] == 4.0:
        data1['STAGE'][i] = 2
 
X = data1.iloc[:,:-1].values
y = data1.iloc[:,-1].values


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
X_train, X_test, y_train, y_test = train_test_split(ennx, enny, test_size = 0.25, random_state = 21)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_te = y_test
y_test = to_categorical(y_test)

###############################################################################
#                                  4. MLP_1                                    #
###############################################################################

trainX = X_train
trainy = y_train
testX = X_test
testy = y_test

# stacked generalization with  meta model 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack
import os
from keras.models import Sequential
from keras.layers import Dense

# fit model on dataset

model= Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(trainX, trainy, epochs=500, verbose=0)
y_pred = model.predict(testX)
ypred_1 = np.argmax(y_pred, axis = 1)
ac = accuracy_score(ypred_1, y_te)
print('Acccuracy: %.3f' % (ac))

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
model1= Sequential()
model1.add(Dense(150, input_dim=11, activation='relu'))
model1.add(Dense(3, activation='sigmoid'))
model1.add(Dense(3, activation='sigmoid'))
model1.add(Dense(3, activation='sigmoid'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model1.fit(trainX, trainy, epochs=500, verbose=0)
y_pred = model.predict(testX)
ypred_1 = np.argmax(y_pred, axis = 1)
ac = accuracy_score(ypred_1, y_te)
print('Acccuracy: %.3f' % (ac))

filename = 'models/model_' + str(1) + '.h5'
model1.save(filename)
print('>Saved %s' % filename)

model2= Sequential()
model2.add(Dense(200, input_dim=11, activation='relu'))
model2.add(Dense(3, activation='sigmoid'))
model2.add(Dense(3, activation='sigmoid'))
model2.add(Dense(3, activation='sigmoid'))
model2.add(Dense(3, activation='sigmoid'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model2.fit(trainX, trainy, epochs=500, verbose=0)
y_pred = model.predict(testX)
ypred_1 = np.argmax(y_pred, axis = 1)
ac = accuracy_score(ypred_1, y_te)
print('Acccuracy: %.3f' % (ac))

filename = 'models/model_' + str(2) + '.h5'
model1.save(filename)
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

###############################################################################
#                                  6. MLP_3                                    #
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

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
		# make prediction
        yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
#    print(stackX.shape)
    return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = GradientBoostingClassifier()
	model.fit(stackedX, inputy)
	return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat


###############################################################################
#                              7. Stacking                                    #
###############################################################################
    
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    y_pred = lb.fit_transform(y_pred)
    return y_pred, y_test


accuracy = []
mcc_score = []
auc_score = []
f_score = []
sensitivity_score = []
specificity_score = []

cv = KFold(n_splits = 10, random_state = 42, shuffle = True)
for train_index, test_index in cv.split(ennx):
    X_train1, X_test1, y_train1, y_test1 = ennx[train_index], ennx[test_index], enny[train_index], enny[test_index]
    model = fit_stacked_model(members, X_train1, y_train1)
    yhat = stacked_prediction(members, model, X_test1)
    
    yhat1, y_test2 = multiclass_roc_auc_score(y_test1, yhat)

    acc = accuracy_score(y_test1, yhat)
    auc = roc_auc_score(y_test2, yhat1)
    f = f1_score(y_test1, yhat, average='macro')
    mcc = matthews_corrcoef(y_test1, yhat)
    cm = confusion_matrix(y_test1, yhat)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    
    accuracy.append(acc)
    auc_score.append(auc)
    f_score.append(f)
    mcc_score.append(mcc)
    sensitivity_score.append(sensitivity)
    specificity_score.append(specificity)
    
print("Accuracy: " + statistics.mean(accuracy).__str__())
print("AUC: " + statistics.mean(auc_score).__str__())
print("F1_Score: " + statistics.mean(f_score).__str__())
print("MCCC: " + statistics.mean(mcc_score).__str__())
print("Sensitivity: " + statistics.mean(sensitivity_score).__str__())
print("Specficity: " + statistics.mean(specificity_score).__str__())

#Acccuracy: 0.842
#>Saved models/model_0.h5
#Acccuracy: 0.842
#>Saved models/model_1.h5
#Acccuracy: 0.842
#>Saved models/model_2.h5
#>loaded models/model_0.h5
#>loaded models/model_1.h5
#>loaded models/model_2.h5
#Loaded 3 models
#Accuracy: 0.9571428571428572
#AUC: 0.9458333333333333
#F1_Score: 0.9496212121212121
#MCCC: 0.9062163891034569
#Sensitivity: 0.925
#Specficity: 0.9666666666666667