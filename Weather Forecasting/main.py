print('run started')

import numpy as np
# import matplotlib.pyplot as plt
# import sklearn as sk
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.neural_network import MLPClassifier
# import sklearn.preprocessing
# import sklearn.impute
import csv                                              # library for using .csv files

# take the data from test.csv and put it in a list called data
# with open('Weather Forecasting/Data/y_100-data-points.csv', newline='') as csvfile:
#     data = list(csv.reader(csvfile))

data = np.genfromtxt('Weather Forecasting/Relevant Data/y_2011-data-points.csv', dtype=float, delimiter=',', skip_header=1)
y = data[:,1] # only relevant stuff; all rows of column 1

# print(y)
# print(y[1:5])
# print(type(y))

# print('y successfully retrieved')

data = np.genfromtxt('Weather Forecasting/Relevant Data/x_2011-data-points.txt', dtype=float, delimiter=',', skip_header=33)
x = data[:,3:] # only relevant stuff; all rows of column 3 till end

# print(y)
# print(x[1:5])
# print(type(x))
# print(x[0:2,:])

# print('y successfully retrieved')



# X = data[:,:-1]
# y = data[:,-1]
