print('run started')

import numpy as np
import sklearn as sk
import sklearn.impute
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import sklearn.preprocessing
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.neural_network import MLPClassifier

try:
    data = np.genfromtxt('Weather Forecasting/Relevant Data/y_100-2011-data-points.csv', 
    dtype=float, delimiter=',', skip_header=1)
    y = data[:,1] # only relevant stuff; all rows of column 1
except:
    print('error in retrieving y')

try:
    data = np.genfromtxt('Weather Forecasting/Relevant Data/x_100-2011-data-points.txt', 
    dtype=float, delimiter=',', skip_header=33)
    X = data[:,3:] # only relevant stuff; all rows of column 3 till end
except:
    print('error in retrieving X')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=None)

# print("shape of X_train: {}".format(X_train.shape))
# print("shape of X_test: {}".format(X_test.shape))
# print("shape of y_train: {}".format(y_train.shape))
# print("shape of y_test: {}".format(y_test.shape))

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

#############################################################################################

# from https://machinelearningmastery.com/make-predictions-scikit-learn/
# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# fit final model
model = LinearRegression()
model.fit(X_train, y_train)
# make a prediction
y_pred = model.predict(X_test)
# show the inputs and predicted outputs
# for i in range(len(X_test)):
# 	print("X=%s, Predicted=%s" % (X_test[i], y_test[i]))

from sklearn.metrics import mean_squared_error
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))