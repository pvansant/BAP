'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: 
'''

import os
# os.system('cls') # clears the command window
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

import functions as fs
import matplotlib.pyplot as plt
import numpy as np
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


# extracting data from csv file
X, y = fs.retrieveGenerationData()



####################################### IMPROVED OG NN

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X = imp.fit_transform(X)

model = keras.models.load_model('forecasting\savedCFModel')
y_pred = model.predict(X)

currentUsage = 0.33
newFeature = np.delete(np.insert(y_pred,0,currentUsage),-1)[:,np.newaxis]
newX = np.append(X, newFeature, axis=1)

# print(y_pred.shape)
# print(newFeature.shape)
# print(X.shape)
# print(newX.shape)

# print(np.mean(y_pred))

# mse_krr = mean_squared_error(y_test, y_pred)

# print('\nThe MSE is', mse_krr)
# print('The RMSE is', mse_krr**0.5)
# print('The relative error is', mse_krr**0.5/30800)


####################################### IMPROVED OG NN

X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=.3, random_state=42)

# checking and handling missing values 
# imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
# X_train = imp.fit_transform(X_train)
# X_test = imp.fit_transform(X_test)


####################################### IMPROVED OG NN

# from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=23, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

epochs = 2
batch_size = 100
verbose = 2         # 0 to show nothing; 1 or 2 to show the progress
n_splits = 2

model = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=verbose)

# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', model))
pipeline = Pipeline(estimators)

### load an existing model 
# model = keras.models.load_model('weatherForecasting/anotherTest')


### make a single model and evaluate it with the test set
MSE = fs.trainWithoutCurve(X_train, y_train, pipeline)

y_pred = pipeline.predict(X_test)
testMSE = mean_squared_error(y_test, y_pred)
testRootMSE = MSE**0.5

### print the results
fs.printTrainingResults(X_train, epochs, batch_size, n_splits, baseline_model, MSE)


### save the model
# model.model.save('weatherForecasting/savedModel')


# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')