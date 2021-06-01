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


import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *

data = np.load('data_V1.npz')
predSun = data['predSun']
predWind = data['predWind']
predDemand = data['predDemand']

from measurements import readMeasurements
from optimizationSetup import modelPredictiveControl
# setup data, Retrieve the measurements (the real data set of k=0), Make predictions using the ANN for demand and generation (for k = 1 till k = 169 (7 days))
time = []
for i in range(169):
    time.append(i)

SoCDiff_ini = readMeasurements(0,len(time),predSun,predWind,predDemand) # one week
#SoCDiff_ini = readMeasurements(0,len(predDemand),predSun,predWind,predDemand) # one year
SoCDiff = {time[i]: SoCDiff_ini[i] for i in range(len(time))}
#print(SoCDiff_ini)

diffsum = 0
for i in range(len(SoCDiff_ini)):
    diffsum = diffsum + SoCDiff_ini[i]
#print(diffsum)

SoCIni = 20.0
SoCRaw = [20.0]

for i in range(len(time)-1):
    SoCRaw.append(SoCRaw[i] + SoCDiff_ini[i])


setPoint_ini = []
for i in range(len(time)):
    setPoint_ini.append(60.0)

setPoint = {time[i]: setPoint_ini[i] for i in range(len(time))}

weight_ini = []
for i in range(len(time)):
    weight_ini.append(float(len(time)-i))

weight = {time[i]: weight_ini[i] for i in range(len(time))}

dCost_ini = []
for i in range(len(time)):
    dCost_ini.append(0.1*weight_ini[i])

dCost = {time[i]: dCost_ini[i] for i in range(len(time))}

dMax = 1

controlLevelIni = 0

# Creathe controller model ones so that it can be solved later
mpc = modelPredictiveControl(time,SoCIni,SoCDiff,setPoint,weight,dCost,dMax,controlLevelIni)
#mpc.pprint() 

# solve the model ones
solver = SolverFactory('glpk') 
solver.solve(mpc, tee = True)

#plot
# plot the estimated SoC over the time horizon
tempSoC = [mpc.SoC[i].value for i in mpc.time]
print(tempSoC)

plt.subplot(3,1,1)
plt.plot(time,tempSoC, c = '#0C7CBA', ls = '-') # plot the SOC
plt.plot(time,setPoint_ini, c = 'black', ls = '--') # plot the set point
plt.plot(time,SoCRaw, c = 'black', ls = '-') # plot the set point
plt.xlabel("Time [hours]")
plt.ylabel("State of Charge [%]")
plt.title("State of charge of the battery over one time horizon")

# plot the optimized control level over the time horizon
tempControlLevel = (mpc.controlLevel[i].value for i in time)
tempControlLevel = list(tempControlLevel)
plt.subplot(3,1,2)
plt.plot(time,tempControlLevel, marker ='o', c = '#0C7CBA', ls ='')
plt.xlabel("Time [hours]")
plt.ylabel("control level")
plt.title("Control level over one time horizon")

SoCDiffRaw = []
for i in range(len(time)):
    SoCDiffRaw.append(tempSoC[i] - SoCRaw[i])
plt.subplot(3,1,3)
plt.plot(time,SoCDiffRaw, c = '#0C7CBA', ls ='-')
plt.xlabel("Time [hours]")
plt.ylabel("\u0394SoC [%]")
plt.title("Difference in SoC due to the controller action")


plt.show()


#loop
    # Retrieve the measurements (the real data set of k=0)

    # Make predictions using the ANN for demand and generation (for k = 1 till k = 169 (7 days))

    # Solve the equations using the pyomo optimizer

    # Implement the first control strategy

    # Move one time step further and loop back

    # Use the wait command (only for implementation)


# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')

