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


X_s, y_s = fs.retrieveSolarData()
X_w, y_w = fs.retrieveWindData()
X_d, y_d = fs.retrieveDemandData()

if sum(np.isnan(y_s))+sum(np.isnan(y_w))+sum(np.isnan(y_d)) != 0: print('nans found')

### scaling
meanSunPower = 12.27*10**6/365/24 # average sun generation per hour
y_s = meanSunPower/np.mean(y_s)*y_s # [Wh/hour] # scale the data

meanWindPower = 10.87*10**6/365/24 # [W] # average wind generation per hour
y_w = meanWindPower/np.mean(y_w)*y_w # [Wh/hour] # scale the data


# splitting data in test and training sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=.3, random_state=42)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=.3, random_state=42)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=.3, random_state=42)

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')

X_s = imp.fit_transform(X_s)
X_train_s = imp.fit_transform(X_train_s)
X_test_s = imp.fit_transform(X_test_s)

X_w = imp.fit_transform(X_w)
X_train_w = imp.fit_transform(X_train_w)
X_test_w = imp.fit_transform(X_test_w)

X_d = imp.fit_transform(X_d)
X_train_d = imp.fit_transform(X_train_d)
X_test_d = imp.fit_transform(X_test_d)

# defining certain variables
verbose = 2         # 0 to show nothing; 1 (much) or 2 (little) to show the progress
n_splits = 5

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
batch_size_s = 200
epochs_s = 500
### wind
batch_size_w = 200
epochs_w = 200
### demand
batch_size_d = 500
epochs_d = 1000

solarModel = KerasRegressor(build_fn=solarBaselineModel, epochs=epochs_s, batch_size=batch_size_s, verbose=verbose)
windModel = KerasRegressor(build_fn=windBaselineModel, epochs=epochs_w, batch_size=batch_size_w, verbose=verbose)
demandModel = KerasRegressor(build_fn=demandBaselineModel, epochs=epochs_d, batch_size=batch_size_d, verbose=verbose)


### save the prediction
# MSE_s = fs.trainWithoutCurve(X_train_s, y_train_s, X_test_s, y_test_s, solarModel)
# y_pred_s = solarModel.predict(X_s)

MSE_w = fs.trainWithoutCurve(X_train_w, y_train_w, X_test_w, y_test_w, windModel)
y_pred_w = windModel.predict(X_w)

# np.save('y_pred_w',y_pred_w)

MSE_d = fs.trainWithoutCurve(X_train_d, y_train_d, X_test_d, y_test_d, demandModel)
y_pred_d = demandModel.predict(X_d)


print('\n############################## SOLAR ##############################\n')
fs.printTrainingResults(X_s, epochs_s, batch_size_s, n_splits, solarBaselineModel, MSE_s)
print('Mean Error as fraction of Maximum:', abs(MSE_s)**0.5/np.max(y_s))

print('\n\n############################## WIND ##############################\n')
fs.printTrainingResults(X_w, epochs_w, batch_size_w, n_splits, windBaselineModel, MSE_w)
print('Mean Error as fraction of Maximum:', abs(MSE_w)**0.5/np.max(y_w))

print('\n\n############################# DEMAND #############################\n')
fs.printTrainingResults(X_d, epochs_d, batch_size_d, n_splits, demandBaselineModel, MSE_d)
print('Mean Error as fraction of Maximum:', abs(MSE_d)**0.5/np.max(y_d))

### mapping
realSolar = y_s
predSolar = y_pred_s
realWind = y_w
predWind = y_pred_w
realDemand = y_d
predDemand = y_pred_d

realWind = realWind[105194-2:113953+1-2]
predWind = predWind[105194-2:113953+1-2]

### saving
# np.savez('dataForControl',
# realSolar = realSolar,
# predSolar = predSolar,
# realWind = realWind,
# predWind = predWind,
# realDemand = realDemand,
# predDemand = predDemand)

print('\n\n############################# TOTALS #############################\n')
print('total generated in a year by realSolar:', np.sum(realSolar)/1000000, 'MWh')
print('total generated in a year by predSolar:', np.sum(predSolar)/1000000, 'MWh')
print('total generated in a year by realWind:', np.sum(realWind)/1000000, 'MWh')
print('total generated in a year by predWind:', np.sum(predWind)/1000000, 'MWh')
print('total generated in a year by realDemand:', np.sum(realDemand)/1000000, 'MWh')
print('total generated in a year by predDemand:', np.sum(predDemand)/1000000, 'MWh')

# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')

from playsound import playsound
playsound('C:/Users/piete/OneDrive/Documents/My Education/06 BAP/Code/sound.wav')
