'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the functionality of the controller for one week is programmed
It used different functions and a loop to control the system.
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
dataForControl = np.load('dataForControl.npz')
predSun = dataForControl['predSolar']
predWind = dataForControl['predWind']
predDemand = dataForControl['predDemand']

# load actual data
Sun = dataForControl['realSolar']
Wind = dataForControl['realWind']
Demand = dataForControl['realDemand']

# Setup input data for the initialization of the model
# Initialize the time array, this represents all hours in the upcoming week.
# This will initialize the Set for the Pyomo model
time = []
for i in range(169):
    time.append(i)

# Initialize all other values and create key parameter pairs
# This will initialize the indexed Parameters for the Pyomo model

# Initialize the difference in state of charge due to the predicted demand and generation
SoCDiff_ini = readPredictions(0,len(time),predSun,predWind,predDemand) # one week
SoCDiff = {time[i]: SoCDiff_ini[i] for i in range(len(time))} # Make it a Dictionary

# Initialize the setpoint the controller tries to reach, this will determine the objective in the Pyomo model
setPoint_ini = []
for i in range(len(time)):
    setPoint_ini.append(60.0) # Is constant, but can be variable
setPoint = {time[i]: setPoint_ini[i] for i in range(len(time))} # Make it a Dictionary

# Initialize the weight for the deviation of the SoC from the setpoint, is used by the objective
weight_ini = []
for i in range(len(time)):
    weight_ini.append(float(len(time)-i)) # The weight decreases linearly over time
weight = {time[i]: weight_ini[i] for i in range(len(time))}# Make it a Dictionary

# Initialize the weight for the change in control signal, is used by the objective
dCost_ini = []
for i in range(len(time)):
    dCost_ini.append(5*weight_ini[i])# The weight decreases linearly over time
dCost = {time[i]: dCost_ini[i] for i in range(len(time))}# Make it a Dictionary

# Initialize the weight for the deviation of the control signal from zero, is used by the objective
cCost_ini = []
for i in range(len(time)):
    cCost_ini.append(2*weight_ini[i])# The weight decreases linearly over time
cCost = {time[i]: cCost_ini[i] for i in range(len(time))}# Make it a Dictionary


# Initialize all other non indexed values
# This will initialize the non indexed Parameters for the Pyomo model
dMax = 1 # The maximum chance allowed in the control signal per hour
SoCIni = 55.0 # The measured SoC at t=0 
controlLevelIni = 0 # The currently employed control signal

# Create controller model, parameters are set to mutable such that the model can be resolved with different parameters
mpc = modelPredictiveControl(time,SoCIni,SoCDiff,setPoint,weight,dCost,cCost,dMax,controlLevelIni)

# Optional line usefull for debugging the controller
#mpc.pprint() 

# Select a solver for solving the model
solver = SolverFactory('glpk') # glpk is a linear solver that can handle discrete values

# Solve the model
solver.solve(mpc, tee = False) # solving the model, tee = true provides extra information on the solution of the solver

# Read values from the model
tempSoC = [mpc.SoC[i].value for i in mpc.time]

# Calculate what happens without the controller
SoCRaw = [SoCIni]
for i in range(len(time)-1):
    SoCRaw.append(SoCRaw[i] + SoCDiff_ini[i])

# plot the SoC over time
plt.subplot(2,1,1)
plt.plot(time,tempSoC, c = '#0C7CBA', ls = '-') # plot the SOC
plt.plot(time,setPoint_ini, c = 'black', ls = '--') # plot the set point
plt.plot(time,SoCRaw, c = 'black', ls = '-') # plot the SoC without the controller
plt.xlabel("Time [hours]")
plt.ylabel("State of Charge [%]")
plt.title("State of charge of the battery over one time horizon")
plt.subplots_adjust(wspace=0.05, hspace=.5)
# plot the optimized control level over the time horizon
tempControlLevel = (mpc.controlLevel[i].value for i in time)
tempControlLevel = list(tempControlLevel)
plt.subplot(2,1,2)
plt.plot(time,tempControlLevel, marker ='o', c = '#0C7CBA', ls ='')
plt.xlabel("Time [hours]")
plt.ylabel("control level")
plt.title("Control level over one time horizon")
# plot the difference when using the controller or not using the controller
#SoCDiffRaw = []
#for i in range(len(time)):
#    SoCDiffRaw.append(tempSoC[i] - SoCRaw[i])
#plt.subplot(3,1,3)
#plt.plot(time,SoCDiffRaw, c = '#0C7CBA', ls ='-')
#plt.xlabel("Time [hours]")
#plt.title("Difference in SoC due to the controller action")

# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')

#show plot
plt.show()