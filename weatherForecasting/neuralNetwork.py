'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, weather data is used to predict the average capacity factor 
of wind turbines in Rotterdam.
'''

import os; os.system('cls') # clears the command window
import datetime
# display a "Run started" message
print("Run started at {}\n".format(datetime.datetime.now().strftime("%X")))

import sklearn.impute
import numpy as np
import sklearn as sk
import functions as f
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor


# extracting data from csv file
try:
    data = np.genfromtxt('weatherForecasting/relevantData/y_2011-data-points.csv', 
    dtype=float, delimiter=',', skip_header=1, skip_footer=12)
    y = data[:,1] # only relevant stuff; all rows of column 1
except:
    print('Error while retrieving y'); exit()

# extracting data from txt file
try:
    data = np.genfromtxt('weatherForecasting/relevantData/x_2011-data-points.txt', 
    dtype=float, delimiter=',', skip_header=33)
    X = data[:,3:] # only relevant stuff; all rows of column 3 till end
except:
    print('Error while retrieving X')
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=None)
# f.printSets(X_train, X_test, y_train, y_test)

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

###################

# load dataset
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))