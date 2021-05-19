
'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the optimization for the mpc is initialized. 
Constants and variables are defined and a model is created.
'''

import os

from numpy.core.function_base import linspace
os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

# Importing libraries
from pyomo.environ import *
from pyomo.dae import *

def modelPredictiveControl(SoC,setPoint):

    #initializing the mpc model using pyomo
    mpc = ConcreteModel()

    # Define Sets
    mpc.time = Set(169) # Define the time steps from 1 to 169 hours

    # Define Parameters
    mpc.SoC = Param(mpc.time, initialize=SoC, mutable=True)
    mpc.setPoint = Param(mpc.time, initialize=setPoint, mutable=True)

    # Define Variables
    mpc.controlLevel = Var()

    # Define Objective functions

    # Define Constraints

    # Return the model
    return mpc



# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')