
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

'''
Inputs:
SoC                     - set of length 169 with the measured SoC at index zero followed by zeros
SoCDiff                 - Set of length 169 with the predicted difference in SoC for the coming hour
setPoint                - Set of length 169 with the Setpoint for every hour
weight                  - Set of length 169 with the weight of the MSE for every hour
dCost                   - Set of length 169 with the weigth penalty for changing the control
dMax                    - integer respresenting the maximum change in the control signal per hour
SoCperControlLevel      - expected change in SoC per control level

output:
Model
'''


def modelPredictiveControl(time,SoC,SoCDiff,setPoint,weight,dCost,dMax,SoCperControlLevel):

    #initializing the mpc model using pyomo
    mpc = ConcreteModel()

    # Define the Set
    mpc.time = Set(initialize= time) # Define the time steps from 0 to 168 hours

    # Define Parameters
    mpc.SoC = Param(mpc.time, initialize=SoC, mutable=True)
    mpc.SoCDiff = Param(mpc.time, initialize=SoCDiff, mutable=True)
    mpc.setPoint = Param(mpc.time, initialize=setPoint, mutable=True)
    mpc.weight = Param(mpc.time, initialize=weight, mutable=True)
    mpc.dCost = Param(mpc.time, initialize=dCost, mutable=True )

    # Define Variables
    mpc.controlLevel = Var(mpc.time, within = Integers, bounds = (-5,5))
    
    # Define Objective functions
    def objrule(mpc):
        return (sum(mpc.weight[i]*(mpc.SoC[i]-mpc.setPoint[i])**2 for i in range(1,len(mpc.time)-1)) + 
        sum(mpc.dCost[i]*(mpc.controlLevel[i]-mpc.controlLevel[i-1])**2 for i in range(1,len(mpc.time)-1)))
    mpc.obj = Objective(expr= objrule, sense = minimize )

    # Define Constraints
    
    def constrDMax(mpc, i):
        constr = 0 
        if i != 0:
            constr = (mpc.controlLevel[i]-mpc.controlLevel[i-1])**2 
        return constr <= dMax 
    mpc.constrDMax = Constraint( mpc.time, rule= constrDMax )

    
    def constrSoC(mpc,i):
        SoCtemp = mpc.SoC[i]
        if i != 0:
            SoCtemp = mpc.SoC[i-1] + mpc.SoCDiff[i-1] + SoCperControlLevel*mpc.controlLevel[i-1]
        return SoCtemp == mpc.SoC[i]
    mpc.constrSoC = Constraint( mpc.time, rule= constrSoC)

    # Return the model
    return mpc



# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')