'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, weather data is used to predict the average capacity factor 
of wind turbines in Rotterdam.
'''

import os; os.system('cls') # clears the command window
import datetime
# display a "Run started" message
print("Run started at {}\n".format(datetime.datetime.now().strftime("%X")))

import numpy as np


# extracting data from txt file
try:
    data = np.genfromtxt('weatherForecasting/relevantData/x_2011-data-points.txt', 
    dtype=float, delimiter=',', skip_header=33)
    X = data[:,[1,2,7,10,21,23]] # only relevant stuff:
    # select YYYYMMDD (col 1; datum), HH (col 2; hour), T (col 7; temperature), 
    # SQ (col 10; sunshine duration), R (col 21; rain), O (col 23; storm)
except:
    print('Error while retrieving X'); exit()

# we want the weeknumber and daynumber instead of the date
timeInfo = np.empty((0,2), int)
for i in X[:,0]:
    # get the date info from the data file
    year = int(str(i)[0:4])
    month = int(str(i)[4:6])
    day = int(str(i)[6:8])
    # make a date from the date info
    time = datetime.datetime(year, month, day)
    # timeInfo will contain the weeknumber (%V) and the daynumber (%u)
    timeInfo = np.append(timeInfo, np.array([[time.strftime("%V"),time.strftime("%u")]]), axis=0)

# the date-column is replaced by a weeknumber and daynumber column
X = np.append(timeInfo, np.delete(X,0,1), 1)

# print(X[0:100,:])

# TODO add output data
# TODO make prediction model