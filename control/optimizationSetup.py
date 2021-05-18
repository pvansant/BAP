'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the optimization for the mpc is initialized. 
Constants and variables are defined.
'''

import os

from numpy.core.function_base import linspace
os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

# Importing libraries
from gekko import GEKKO
import matplotlib.pyplot as plt
import numpy as np
from random import random



#initializing the mpc model using GEKKO
mpc = GEKKO()
mpc.time = linspace(0,7,7*24+1)  #Define the time horizon set at 7 days with time steps of 1 hour

# Parameters (constants and variables that can not be changed by the program)
# c = mpc.Const(value, [name]) or c = value     the constant remains constant over time
# p = mpc.Param([value], [name])                the parameter can change over time often a np array

# Manipulated variables (MV)
# m = mpc.MV([value], [lb], [ub], [integer], [name])
ctrlLevel = mpc.MV(value = 0, lb = -5, ub = 5, integer = True) # initialize the control level and making it discrete
ctrlLevel.STATUS = 1 # allow optimizer to chance the value
ctrlLevel.DMAX = 1 # maximum change per time step for the control level
ctrlLevel.DCOST = 0.1 # inreases cost for changing the value

# Controlled variables (CV)
# c = mpc.CV([value], [lb], [ub], [integer], [name])
SoC= mpc.CV(value=75) # initialize the battery state of charge in percentage # (, ub=100, lb=40)
SoC.STATUS = 1 # Add the setpoint to the objective
mpc.options.CV_TYPE = 1 # 1 = deadband linear, 2 = mean squared error

SoC.SPHI(value = 85)
SoC.SPLO(value = 70)

# Process model
# m.Equation(equation)

mpc.options.IMODE = 6



# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')