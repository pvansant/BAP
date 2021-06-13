'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the functionality of the controller is programmed
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
sun = dataForControl['realSolar']
wind = dataForControl['realWind']
demand = dataForControl['realDemand']

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
    weight_ini.append(1/(1+i)) # The weight decreases linearly over time
weight = {time[i]: weight_ini[i] for i in range(len(time))}# Make it a Dictionary

# Initialize the weight for the change in control signal, is used by the objective
dCost_ini = []
for i in range(len(time)):
    dCost_ini.append(0.1*weight_ini[i])# The weight decreases linearly over time
dCost = {time[i]: dCost_ini[i] for i in range(len(time))}# Make it a Dictionary

# Initialize the weight for the deviation of the control signal from zero, is used by the objective
cCost_ini = []
for i in range(len(time)):
    cCost_ini.append(0.1*weight_ini[i])# The weight decreases linearly over time
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

# prepare lists for plotting
# general
timePlot = [0]
setPointPlot = [60.0]
# With controller
SoCPlot = [SoCIni]
controlLevelPlot = [controlLevelIni]
gridPowerGivePlot = [0]
gridPowerTakePlot = [0]
# without controller
SoCRawPlot = [SoCIni]
gridPowerGiveRawPlot = [0]
gridPowerTakeRawPlot = [0]

#loop
# optimize for the coming week every hour
for t in range(1,500):
    #update control level  
    controlLevelIni = mpc.controlLevel[1].value

    # Retrieve the measurements when using the controller
    prevControl = determineControlSoC(t-1, mpc.controlLevel[0].value)
    if (SoCPlot[-1] + readMeasurement(t,sun,wind,demand) + prevControl > 100):
        mpc.SoCIni.value= SoCIni = 100
        gridPowerGive = abs(SoCPlot[-1] + readMeasurement(t,sun,wind,demand) + prevControl - 100)
        gridPowerTake = 0
    elif (SoCPlot[-1] + readMeasurement(t,sun,wind,demand) + prevControl < 10):
        mpc.SoCIni.value= SoCIni = 10
        gridPowerGive = 0
        gridPowerTake = abs(10 - (SoCPlot[-1] + readMeasurement(t,sun,wind,demand) + prevControl))
    else:
        mpc.SoCIni.value= SoCIni = SoCPlot[-1] + readMeasurement(t,sun,wind,demand) + prevControl
        gridPowerGive = 0
        gridPowerTake = 0

    # Retrieve the measurements without using the controller
    if SoCRawPlot[-1] + readMeasurement(t,sun,wind,demand) >100:
        SoCRaw = 100
        gridPowerGiveRaw = abs(SoCPlot[-1] + readMeasurement(t,sun,wind,demand) - 100)
        gridPowerTakeRaw = 0
    elif (SoCPlot[-1] + readMeasurement(t,sun,wind,demand)< 10):
        SoCRaw = 10
        gridPowerGiveRaw = 0
        gridPowerTakeRaw = abs(10 - (SoCPlot[-1] + readMeasurement(t,sun,wind,demand) ))
    else:
        SoCRaw = SoCPlot[-1] + readMeasurement(t,sun,wind,demand)
        gridPowerGiveRaw = 0
        gridPowerTakeRaw = 0

    # Make predictions using the ANN for demand and generation and prep for the model
    SoCDiff_ini = readPredictions(t,len(time),predSun,predWind,predDemand)
    mpc.SoCDiff[i].value = [SoCDiff_ini[i] for i in range(len(time))] 
    #SoCDiff = {time[i]: SoCDiff_ini[i] for i in range(len(time))}   
    mpc.controlLevelIni.value = controlLevelIni
    # Solve the pyomo optimizer model
    # mpc = modelPredictiveControl(time,SoCIni,SoCDiff,setPoint,weight,dCost,cCost,dMax,controlLevelIni)
    solver.solve(mpc, tee = False)

    # update plot values for both with and without controller
    timePlot.append(t)
    SoCPlot.append(SoCIni)
    setPointPlot.append(60.0)
    controlLevelPlot.append(controlLevelIni)
    gridPowerGivePlot.append(gridPowerGive)
    gridPowerTakePlot.append(gridPowerTake)
    SoCRawPlot.append(SoCRaw)
    gridPowerGiveRawPlot.append(gridPowerGiveRaw)
    gridPowerTakeRawPlot.append(gridPowerTakeRaw)

    # print confirmation every 25 cycles
    if t%10 == 0:
        print('cycle: ', t , ' is done')
    
    # loop back and repeat for the next hours

#plot relevant data for both with and without controller to compare the effect of the controller
print('\n------ With controller ------\n')
print('power delivered to grid:{:.2f}Wh'.format(sum(gridPowerGivePlot)))
k=0
for i in gridPowerGivePlot: 
    if i>0: 
        k += 1
print('Hours of power delivered to grid: {} hours, which is {:.2f} %'.format(k,k/len(gridPowerGivePlot)*100))
print('power taken from grid:{:.2f}Wh'.format(sum(gridPowerTakePlot)))
k=0
for i in gridPowerTakePlot: 
    if i>0: 
        k += 1
print('Hours of power taken from grid: {} hours, which is {:.2f} %'.format(k,k/len(gridPowerGivePlot)*100))

print('\n------ Without controller ------\n')
print('power delivered to grid:{:.2f}Wh'.format(sum(gridPowerGiveRawPlot)))
k=0
for i in gridPowerGiveRawPlot: 
    if i>0: 
        k += 1
print('Hours of power delivered to grid: {} hours, which is {:.2f} %'.format(k,k/len(gridPowerGiveRawPlot)*100))
print('power taken from grid:{:.2f}Wh'.format(sum(gridPowerTakeRawPlot)))
k=0
for i in gridPowerTakeRawPlot: 
    if i>0: 
        k += 1
print('Hours of power taken from grid: {} hours, which is {:.2f} %'.format(k,k/len(gridPowerGiveRawPlot)*100))

# Plot SoC over time
plt.subplot(2,1,1)
plt.plot(timePlot,SoCPlot, c = '#0C7CBA', ls = '-') # plot the SOC
plt.plot(timePlot,setPointPlot, c = 'black', ls = '--') # plot the set point
plt.plot(timePlot,SoCRawPlot, c = 'black', ls = '-') # plot the SoC without the controller
plt.xlabel("Time [hours]")
plt.ylabel("State of Charge [%]")
plt.title("State of charge of the battery over one time horizon")

# plot the optimized control level over the time 
plt.subplot(2,1,2)
plt.plot(timePlot,controlLevelPlot, marker ='o', c = '#0C7CBA', ls ='')
plt.xlabel("Time [hours]")
plt.ylabel("control level")
plt.title("Control level over one time horizon")

# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')

# show plot
plt.show()





