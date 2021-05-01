print('run started')

import numpy as np
import sklearn as sk
import sklearn.impute
from sklearn.model_selection import train_test_split

try:
    data = np.genfromtxt('Weather Forecasting/Relevant Data/x_2011-data-points.txt', 
    dtype=float, delimiter=',', skip_header=33)
    X = data[:,1:] # only relevant stuff; all rows of column 3 till end
    # select YYYYMMDD (col 1; datum), HH (col 2; hour), T (col 7; temperature), SQ (col 10; sunshine duration), R (col 21; rain), O (col 23; storm)
except:
    print('error in retrieving X')
    exit()

# TODO extract more information from the datum (YYYYMMDD)
# print(X[0:4,:])