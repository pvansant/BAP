import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
import sklearn.impute
import csv # library for using .csv files

# take the data from test.csv and put it in a list called data
with open('Weather Forecasting/Data/test.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

# print(data)
# print(type(data))

## simpler thing that kinda does the same
# f = open("Weather Forecasting/Data/test.txt", "r")
# print(f.read())
# f.close()



print('success')