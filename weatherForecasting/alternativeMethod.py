# TODO header

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

########################################################################

# from sklearn.neural_network import MLPRegressor
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
# regr.predict(X_test[:2])


# # from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# # X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# clf.predict(X_test[:2, :])

# clf.predict_proba(X_test[:2, :])

# print("score: {}".format(clf.score(X_test, y_test)))


# print("Regression Score: {}".format(regr.score(X_test, y_test)))

# from sklearn.metrics import mean_squared_error
# print("MSE: {}".format(mean_squared_error(y_test, y_pred)))

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# X, y = make_regression(n_samples=200, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
regr.predict(X_test)

print("Score: {}".format(regr.score(X_test, y_test)))