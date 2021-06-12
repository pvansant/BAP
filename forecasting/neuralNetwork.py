'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, weather data is used to predict the average capacity factor of wind turbines in Rotterdam. This is done using an Artificial Neural Network.
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


# suppress depreciation warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

X, y = fs.retrieveSolarData()
# X, y = fs.retrieveWindData()
# X, y = fs.retrieveDemandData()

# print('X.shape:', X.shape)
# print('y.shape:', y.shape)
# print('X:\n', X)
# print('y:\n', y)


### check for nans present
# k = 0
# for i in X.flatten(): 
#     if math.isnan(i): print('error in X at position ', k)
#     k+=1
# k = 0
# for i in y: 
#     if math.isnan(i): print('error in y at position ', k)
#     k+=1

# splitting data in test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
# fs.printSets(X_train, X_test, y_train, y_test) # enable to print set shapes

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X = imp.fit_transform(X)
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

### check for nans present
k = 0
for i in y: 
    if math.isnan(i): print('error in X at position ', k)
    k+=1

# defining certain variables
verbose = 0         # 0 to show nothing; 1 or 2 to show the progress
n_splits = 10

# from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

# define solar base model
def solarBaselineModel():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# define generation base model
def windBaselineModel():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# define demand base model
def demandBaselineModel():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

### solar
batch_size = 200
epochs = 500
### wind
# batch_size = 200
# epochs = 200
### demand
# batch_size = 500
# epochs = 1000

model = KerasRegressor(build_fn=solarBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)
# model = KerasRegressor(build_fn=windBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)
# model = KerasRegressor(build_fn=demandBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', model))
pipeline = Pipeline(estimators)

### load an existing model
# model = keras.models.load_model('weatherForecasting/anotherTest')


### make a single model and evaluate it with the test set
# MSE = fs.trainWithoutCurve(X_train, y_train, model)

# y_pred = model.predict(X_test)
# testMSE = mean_squared_error(y_test, y_pred)
# testRootMSE = MSE**0.5
# print('test rmse',testRootMSE)
# print('test relative rmse',testRootMSE/30800)


### make multiple models using cross_val_score and evaluate it using validation sets from the training set
# MSE, STD = fs.performCrossValidation(X_train, y_train, n_splits, pipeline)


### print the results
# fs.printTrainingResults(X_train, epochs, batch_size, n_splits, solarBaselineModel, MSE)
# print('for demand; relative root mse:',MSE**0.5/30800)


### make a single model (without the pipeline) and show the learning curve
# fs.trainWithCurve(X_train, y_train, model)


### save the model
# model.model.save('weatherForecasting/savedModel')


### save the prediction
# MSE = fs.trainWithoutCurve(X_train, y_train, model)
# y_pred = model.predict(X)
# np.save('predictedDemand_V1', y_pred)




# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')