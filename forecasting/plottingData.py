'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the functionality of the controller is programmed
It used different functions and a loop to control the system.
'''

import os

from numpy.core.function_base import linspace
from numpy.lib.function_base import append, diff
# os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

# import libraries and python functions
import numpy as np
import matplotlib.pyplot as plt
# from pyomo.environ import *
# from pyomo.dae import *
# from measurements import readMeasurement
# from measurements import readPredictions
# from measurements import determineControlSoC
# from optimizationSetup import modelPredictiveControl

# # load predictions 
# data = np.load('data_V2.npz')
# predSun = data['predSun']
# predWind = data['predWind']
# predDemand = data['predDemand']

# # load actual data
# data1 = np.load('data_original_V2_with-scaling.npz')
# sun = predSun
# wind = data1['windOutput']
# demand = data1['demandOutput']

dataForControl = np.load('dataForControl_with-previous-hour.npz')
realSolar =     dataForControl['realSolar']
predSolar =     dataForControl['predSolar']
realWind =      dataForControl['realWind']
predWind =      dataForControl['predWind']
realDemand =    dataForControl['realDemand']
predDemand =    dataForControl['predDemand']

plt.subplot(3,1,1)
plt.plot(realSolar, c = 'black', ls = '-') # plot the set point
plt.plot(predSolar, c = '#0C7CBA', ls = '-') # plot the SOC

plt.xlabel("Time [hours]")
plt.ylabel("predicted solar generation [Wh]")
plt.subplot(3,1,2)
plt.plot(realWind, c = 'black', ls = '-') # plot the set point
plt.plot(predWind, c = '#0C7CBA', ls = '-') # plot the SOC

plt.xlabel("Time [hours]")
plt.ylabel("predicted wind generation [Wh]")
plt.subplot(3,1,3)
plt.plot(realDemand, c = 'black', ls = '-') # plot the set point
plt.plot(predDemand, c = '#0C7CBA', ls = '-') # plot the SOC

plt.xlabel("Time [hours]")
plt.ylabel("predicted demand [Wh]")

plt.show()


# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')

