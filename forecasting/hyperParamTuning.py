'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: With this script, the hyperparameters of the ANN for a certain data set can tuned.
'''

import os
# os.system('cls') # clears the command window
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('\nRun started at', start_time.strftime("%X"), '\n')

import matplotlib.pyplot as plt
import numpy as np
import functions as fs
import sklearn as sk
import sklearn.impute
import math
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from playsound import playsound


# suppress depreciation warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


# retrieve data of one of these three
# X, y = fs.retrieveSolarData()
# X, y = fs.retrieveWindData()
# X, y = fs.retrieveDemandData()


# splitting data in test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
fs.printSets(X_train, X_test, y_train, y_test) # enable to print set shapes

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X = imp.fit_transform(X)
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)


# defining certain variables
epochs = 1000
batch_size = 500
verbose = 2         # 0 to show nothing; 1 or 2 to show the progress
n_splits = 2

# from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

# define solar base model
def solarBaselineModel(neurons1=1, neurons2=1, neurons3=1):
    # create model
    model = Sequential()

    model.add(Dense(neurons1, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons3, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# define generation base model
def windBaselineModel(neurons1=1, neurons2=1, neurons3=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons1, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# define demand base model
def demandBaselineModel(neurons1=1, neurons2= 1, neurons3=1, neurons4=1, neurons5=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons1, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# create the model of one of these three
# model = KerasRegressor(build_fn=solarBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)
# model = KerasRegressor(build_fn=windBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)
# model = KerasRegressor(build_fn=demandBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)


# define the grid search parameters
batch_size = [200, 500, 1000]
epochs = [10, 25, 50, 100, 200, 500, 1000, 2000]

neurons1 = [150, 300]
neurons2 = [100, 200]
neurons3 = [50, 100]
neurons4 = [25, 50]
neurons5 = [10, 20]

param_grid = dict(batch_size=batch_size, epochs=epochs, neurons1=neurons1, neurons2=neurons2, neurons3=neurons3, neurons4=neurons4, neurons5=neurons5)


# evaluate all combinations using 3-fold cross validation
# from https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')
playsound('C:/Users/piete/OneDrive/Documents/My Education/06 BAP/Code/sound.wav')