'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, data of the predictions and actual measurements is plotted.
'''

import os
from numpy.core.function_base import linspace
from numpy.lib.function_base import append, diff
os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

# import libraries and python functions
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
from measurements import readMeasurement
from measurements import readPredictions
from measurements import determineControlSoC
from optimizationSetup import modelPredictiveControl

# load predictions 
data = np.load('data_V2.npz')
predSun = data['predSun']
predWind = data['predWind']
predDemand = data['predDemand']

# load actual data
data1 = np.load('data_original_V3.npz')
sun = predSun
wind = data1['windOutput']
demand = data1['demandOutput']

plt.subplot(3,1,1)
plt.plot(sun, c = 'black', ls = '-') # plot the set point
plt.plot(predSun, c = '#0C7CBA', ls = '-') # plot the SOC
plt.title("Solar generation data")
plt.xlabel("Time [hours]")
plt.ylabel("predicted solar generation [Wh]")
plt.subplot(3,1,2)
plt.plot(wind, c = 'black', ls = '-') # plot the set point
plt.plot(predWind, c = '#0C7CBA', ls = '-') # plot the SOC
plt.title("Wind generation data")
plt.xlabel("Time [hours]")
plt.ylabel("predicted wind generation [Wh]")
plt.subplot(3,1,3)
plt.plot(demand, c = 'black', ls = '-') # plot the set point
plt.plot(predDemand, c = '#0C7CBA', ls = '-') # plot the SOC
plt.title("Demand data")
plt.xlabel("Time [hours]")
plt.ylabel("predicted demand [Wh]")
plt.subplots_adjust(wspace=0.05, hspace=1.5)
# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')

# show plot

plt.show()