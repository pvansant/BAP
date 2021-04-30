print('run started')

import numpy as np
# import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.neural_network import MLPClassifier
# import sklearn.preprocessing
import sklearn.impute
# import csv                                              # library for using .csv files

# take the data from test.csv and put it in a list called data
# with open('Weather Forecasting/Data/y_100-data-points.csv', newline='') as csvfile:
#     data = list(csv.reader(csvfile))

data = np.genfromtxt('Weather Forecasting/Relevant Data/y_100-2011-data-points.csv', dtype=float, delimiter=',', skip_header=1)
y = data[:,1] # only relevant stuff; all rows of column 1
print('y successfully retrieved')
    # print(y)
    # print(y[1:5])
    # print(type(y))

data = np.genfromtxt('Weather Forecasting/Relevant Data/x_100-2011-data-points.txt', dtype=float, delimiter=',', skip_header=33)
X = data[:,3:] # only relevant stuff; all rows of column 3 till end
print('X successfully retrieved')
    # print(y)
    # print(x[1:5])
    # print(type(x))
    # print(x[0:2,:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=None)
# X_train = X[]

print("shape of X_train: {}".format(X_train.shape))
print("shape of X_test: {}".format(X_test.shape))
print("shape of y_train: {}".format(y_train.shape))
print("shape of y_test: {}".format(y_test.shape))

# standardization
# scaler = sk.preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

#########################################################################################################

# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
# X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
model = LinearRegression()
model.fit(X_train, y_train)
# new instances where we do not know the answer
# Xnew, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
# make a prediction
y_pred = model.predict(X_test)
# show the inputs and predicted outputs
# for i in range(len(X_test)):
# 	print("X=%s, Predicted=%s" % (X_test[i], y_test[i]))
# for i in range(3):
# 	print("X=%s, Predicted=%s" % (X_test[i], y_test[i]))

# print(model.score(X_test, y_pred))

# from sklearn.metrics import balanced_accuracy_score
# print(balanced_accuracy_score(y_test, y_pred))

# from sklearn.metrics import mean_absolute_percentage_error
# mean_absolute_percentage_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))