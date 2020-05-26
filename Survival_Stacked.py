import pandas as pd
import numpy as np
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

data2 = pd.read_csv('main_data.csv')
data2 = data2.drop(['Unnamed: 0'], axis = 1)

X = data2.iloc[:, :-1]
y = data2.iloc[:, -1]
y.value_counts()

###############################################################################
#                        2. Scale Data                                         #                                      
###############################################################################

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

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
#                                  4. MLP_1                                    #
###############################################################################

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 50, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 24))

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
                     activation = 'relu', input_dim =24))

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
#                                  6. MLP_3                                   #
###############################################################################


model2 = Sequential()
model2.add(Dense(units = 200, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 24))

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
for train_index, test_index in cv.split(X):
    X_train1, X_test1, y_train1, y_test1 = X[train_index], X[test_index], y[train_index], y[test_index]

    model = fit_stacked_model(members, X_train1, y_train1)
    yhat = stacked_prediction(members, model, X_test1)

    acc = accuracy_score(y_test1, yhat)
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
#ENN
#Model Accuracy: 0.920
#Model Accuracy: 0.920
#Model Accuracy: 0.940
#Accuracy: 0.98
#AUC: 0.9756862745098039
#f1: 0.9596235483155803
#MCCC: 0.9780934343434343
#Sensitivity: 0.990909090909091
#Specficity: 0.9652777777777778
#
#Model Accuracy: 0.775
#Model Accuracy: 0.721
#Model Accuracy: 0.730
#Accuracy: 0.8847474747474747
#AUC: 0.8938150076873481
#f1: 0.771612374246195
#MCCC: 0.8864485889777651
#Sensitivity: 0.8631156696088046
#Specficity: 0.9097815083467258

#########################MLP##################################

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(25, input_dim=24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(trainX, trainy, epochs=100, verbose=0)
y_pred = model.predict(testX)
y_pred = np.round(y_pred)

from sklearn.metrics import classification_report,accuracy_score
cr = classification_report(y_pred, testy)
ac = accuracy_score(y_pred, testy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, testy)

from sklearn.metrics import accuracy_score
ac_ann = accuracy_score(y_pred, testy)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_pred, testy)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_pred, testy)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(testy, y_pred)
print('AUC: %.3f' % auc)

#ac=76.4
###############################GBDT######################


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
if __name__ == '__main__':

    kf = KFold(n_splits=8, shuffle = True, random_state = 42)
    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
gbrt = GradientBoostingClassifier(max_depth= None, n_estimators=150, learning_rate=1, random_state=42)
gbrt.fit(X_train, y_train)
predicted_y.extend(gbrt.predict(X_test))
expected_y.extend(y_test)
accuracy = metrics.accuracy_score(expected_y, predicted_y)
print("Accuracy: " + accuracy.__str__())

   
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( expected_y,  predicted_y)

from sklearn.metrics import accuracy_score
ac = accuracy_score( expected_y,  predicted_y)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score( expected_y,  predicted_y)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef( expected_y,  predicted_y)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(expected_y,  predicted_y)
print('AUC: %.3f' % auc)
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(expected_y,  predicted_y)
print(ac)
print(f1)
print(mcc)

##########################Random_Forests####################

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200,random_state=100)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

from sklearn.metrics import accuracy_score
ac_rf = accuracy_score(y_test, y_pred)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
print('AUC: %.3f' % auc)
#76