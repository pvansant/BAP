'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the functionality of the controller is programmed
It used different functions and a loop to control the system.
'''

import os

from numpy.core.function_base import linspace
from numpy.lib.function_base import append
os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')


import optimizationSetup
import numpy as np
import matplotlib.pyplot as plt


# setup data, Retrieve the measurements (the real data set of k=0), Make predictions using the ANN for demand and generation (for k = 1 till k = 169 (7 days))
time = []
for i in range(169):
    time.append(i)

SoC_ini = [55]
for i in range(168):
    SoC_ini.append(0)

SoC = {time[i]: SoC_ini[i] for i in range(len(time))}

# SoCDiff
#SoCDiff = {time[i]: SoCDiff_ini[i] for i in range(len(time))}

setPoint_ini = []
for i in range(169):
    setPoint_ini.append(55)

setPoint = {time[i]: setPoint_ini[i] for i in range(len(time))}

weight_ini = []
for i in range(169):
    weight_ini.append(1/(i+1))

weight = {time[i]: weight_ini[i] for i in range(len(time))}

dCost_ini = []
for i in range(169):
    dCost_ini.append(0.1*weight_ini[i])

dCost = {time[i]: dCost_ini[i] for i in range(len(time))}

dMax = 1

controlSoC_ini = []
for i in range(169):
    controlSoC_ini.append(0)

controlSoC = {time[i]: controlSoC_ini[i] for i in range(len(time))}
# optimizer setup
#mpc = modelPredictiveControl(time,SoC,SoCDiff,setPoint,weight,dCost,dMax,controlSoC)

# run 
#solver = SolverFactory('ipopt') 
#solver.solve(mpc)

#plot
tempSoC = (mpc.SoC[i] for i in mpc.time)
tempSoC = np.array(tempSoC)
plt.subplot(2,1,1)
plt.plot(np.array(time),np.array(tempSoC), c = '#0C7CBA', ls = '-')
plt.plot(np.array(time),np.array(setPoint), c = 'black', ls = '--')
plt.xlabel("Time [hours]")
plt.ylabel("State of Charge [%]")
plt.title("State of charge of the battery over one time horizon")

controlLevel = (mpc.controlLevel[i] for i in mpc.time)
controlLevel = np.array(controlLevel)
plt.subplot(2,1,2)
plt.plot(np.array(time),np.array(controlLevel), marker ='o', c = '#0C7CBA', ls ='')
plt.xlabel("Time [hours]")
plt.ylabel("control level")
plt.title("Control signals over one time horizon")

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




