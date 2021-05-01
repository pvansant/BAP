# TODO better file name
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
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
f.printSets(X_train, X_test, y_train, y_test)

# TODO standardization

# checking and handling missing values 
# FIXME improve NaN handling
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

# TODO feature selection

# create model
model = LinearRegression()
model.fit(X_train, y_train)

# make a prediction
y_pred = model.predict(X_test)

print("The MSE of this model is: {}".format(mean_squared_error(y_test, y_pred)))