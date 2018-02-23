import keras
import numpy as np
import pandas as pd

dataset_X= pd.read_csv('train\X_train.txt',delim_whitespace=True, index_col=False, header=None)
dataset_Y=pd.read_csv('train\y_train.txt', header=None)

dataset = pd.concat([dataset_X, dataset_Y],axis = 1)

from sklearn.utils import shuffle
dataset = shuffle(dataset)

X_train = dataset.iloc[:,0:561].values
Y_train = dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu', input_dim = 561))
#classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(units = 561, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 16, epochs = 32)


dT_X=pd.read_csv('test\X_test.txt', delim_whitespace=True, index_col=False, header=None)
X_given=dT_X.iloc[:,:].values

dT_Y=pd.read_csv('test\Y_test.txt',header=None)
Y_given=dT_Y.iloc[:,:].values

X_given=sc_X.transform(X_given)

preds = classifier.predict(X_given)
final_preds=[]
for i in range(0,len(preds)) :
    m=max(preds[i])
    for j in range(0,7) :
        if preds[i][j]==m :
            final_preds.append(j)

final_preds = np.array(final_preds)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_given,final_preds)

#EVALUATING AND TUNING ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units =250,kernel_initializer = 'uniform', activation = 'relu', input_dim = 561))
    classifier.add(Dense(units =250, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units =561, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units =7, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 16, epochs = 32)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [8,10, 16,25, 32],
              'epochs': [24, 32, 50]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = None,
                           cv = 2)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

