'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, weather data is used to predict the average capacity factor of wind turbines in Rotterdam. This is done using an Artificial Neural Network.
'''

import os
# os.system('cls') # clears the command window
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('\nRun started at ', start_time.strftime("%X"), '\n')

import matplotlib.pyplot as plt
import numpy as np
import functions as fs
import sklearn as sk
import sklearn.impute
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# suppress depreciation warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# X, y = fs.retrieveWindData()
X, y = fs.retrieveDemandData()


# splitting data in test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
# fs.printSets(X_train, X_test, y_train, y_test) # enable to print set shapes

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

# defining certain variables
epochs = 200
batch_size = 100
verbose = 0         # 0 to show nothing; 1 or 2 to show the progress
n_splits = 2

# from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# define generation base model
def windBaselineModel():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# define demand base model
def demandBaselineModel():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# model = KerasRegressor(build_fn=generationBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)
model = KerasRegressor(build_fn=demandBaselineModel, epochs=epochs, batch_size=batch_size, verbose=verbose)


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', model))
pipeline = Pipeline(estimators)

### load an existing model
# model = keras.models.load_model('weatherForecasting/anotherTest')


### make a single model and evaluate it with the test set
# MSE = fs.trainWithoutCurve(X_train, y_train, pipeline)

# y_pred = pipeline.predict(X_test)
# testMSE = mean_squared_error(y_test, y_pred)
# testRootMSE = MSE**0.5
# print('test rmse',testRootMSE)
# print('test relative rmse',testRootMSE/30800)


### make multiple models using cross_val_score and evaluate it using validation sets from the training set
# MSE, STD = fs.performCrossValidation(X_train, y_train, n_splits, pipeline)


### print the results
# fs.printTrainingResults(X_train, epochs, batch_size, n_splits, windBaselineModel, MSE)
# print('for demand; relative root mse:',MSE**0.5/30800)


### make a single model (without the pipeline) and show the learning curve
# fs.trainWithCurve(X_train, y_train, model)


### save the model
# model.model.save('weatherForecasting/savedModel')

# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')