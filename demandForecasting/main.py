'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: # TODO description
'''

import os; # os.system('cls') # clears the command window
import datetime
# display a "Run started" message
print("Run started at {}\n".format(datetime.datetime.now().strftime("%X")))

import sklearn as sk
import numpy as np
import numpy.ma as ma


# extracting input data from txt file
try:
    data = np.genfromtxt('weatherForecasting/relevantData/x_2011-data-points.txt', 
    dtype=float, delimiter=',', skip_header=33)
    X = data[:,[1,2,7,10,21,23]] # only relevant stuff:
    # select YYYYMMDD (col 1; datum), HH (col 2; hour), T (col 7; temperature), 
    # SQ (col 10; sunshine duration), R (col 21; rain), O (col 23; storm)
except:
    print('Error while retrieving input data'); exit()

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

# print(X[0:100,:]) # FIXME gekke temperaturen

# NOTE these datasets are of houses of which some have their own generation

# TODO make prediction model

# extracting output data from csv file
try:
    data = np.genfromtxt('demandForecasting/relevantData/Zonnedael - slimme meter dataset - 2013 - Levering.csv', 
    dtype=float, delimiter=',', skip_header=1, skip_footer=34992)
    y = data[:,2:-2] # only relevant stuff:
    # select YYYYMMDD (col 1; datum), HH (col 2; hour), T (col 7; temperature), 
    # SQ (col 10; sunshine duration), R (col 21; rain), O (col 23; storm)
except:
    print('Error while retrieving output data'); exit()

y = y.reshape(-1,4,y.shape[-1]).sum(1)

# from https://stackoverflow.com/q/18689235
# FIXME using pandas
y = np.where(np.isnan(y), ma.array(y, mask=np.isnan(y)).mean(axis=1)[:, np.newaxis], y)
y = y.sum(1)