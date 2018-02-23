# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:33:20 2018

@author: Vaibhav
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_X= pd.read_csv('train\X_train.txt',delim_whitespace=True, index_col=False, header=None)
X=dataset_X.iloc[:,:].values

dataset_Y=pd.read_csv('train\y_train.txt', header=None)
Y=dataset_Y.iloc[:,:].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.01,random_state=0,shuffle=True)

#NO FEATURE SCALING FOR DECISION TREES AS ITS NOT BASED ON EUCLIDEAN DISTANCES
#BUT WILL DO IT FOR BETTER PLOTTING THE DECISION BOUNDARY

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=350,criterion='entropy',random_state=0, oob_score=True)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

accuracy=classifier.oob_score_

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [200, 250, 300, 350], 'criterion': ['entropy']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Testing on given test set

dT_X=pd.read_csv('test\X_test.txt', delim_whitespace=True, index_col=False, header=None)
X_given=dT_X.iloc[:,:].values

dT_Y=pd.read_csv('test\Y_test.txt',header=None)
Y_given=dT_Y.iloc[:,:].values

X_given=sc_x.transform(X_given)
Y_pred_on_given=classifier.predict(X_given)

realcm=confusion_matrix(Y_given,Y_pred_on_given)
final_accuracy=classifier.oob_score_

