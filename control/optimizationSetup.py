
'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the optimization for the mpc is initialized. 
Parameters and variables are defined and a model is created.
'''
import os
# Importing libraries
from pyomo.environ import *
from pyomo.dae import *

'''
function: create and initialize the optimization for the mpc
Inputs:
time                    - list of length 169 representing every hour in the coming week
SoCIni                  - integer respresenting the SoC at t = 0
SoCDiff                 - List of length 169 with the predicted difference in time
setPoint                - Set of length 169 with the Setpoint for every hour
weight                  - Set of length 169 with the associated weight for deviation from the setpoint for every hour
dCost                   - Set of length 169 with the associated weight penalty for changing the control level
cCost                   - Set of length 169 with the associated weight for deviation of the control level from level zero
dMax                    - integer respresenting the maximum change in the control signal per hour
controlLevelIni         - integer respresenting current control level
output:
Mpc                     - the model for the optimizer of the model predictive controller
'''

def modelPredictiveControl(time,SoCIni,SoCDiff,setPoint,weight,dCost,cCost,dMax,controlLevelIni):

    #initializing the mpc model using pyomo
    mpc = ConcreteModel()

    # Define the Set
    mpc.time = Set(initialize= time) # Define the time steps from 0 to 168 hours
   
    # Define Parameters   
    mpc.SoCDiff = Param(mpc.time, initialize=SoCDiff, mutable=True, within = Any)
    mpc.setPoint = Param(mpc.time, initialize=setPoint, mutable=True)
    mpc.weight = Param(mpc.time, initialize=weight, mutable=True)
    mpc.dCost = Param(mpc.time, initialize=dCost, mutable=True )
    mpc.cCost = Param(mpc.time, initialize=cCost, mutable=True )
    mpc.SoCIni = Param(initialize=SoCIni, mutable = True)
    mpc.controlLevelIni = Param(initialize=controlLevelIni, mutable = True)
    mpc.dMax = Param(initialize=dMax, mutable = True)
    
    # Define Variables
    mpc.controlLevel = Var(mpc.time, within = Integers, bounds = (0,4))
    mpc.SoC = Var(mpc.time, within = Reals)
    mpc.deltaSetPointPos = Var(mpc.time , within = NonNegativeReals)
    mpc.deltaSetPointNeg = Var(mpc.time , within = NonNegativeReals)
    mpc.controlLevelPos = Var(mpc.time , within = NonNegativeIntegers)
    mpc.controlLevelNeg = Var(mpc.time , within = NonNegativeReals)

    # Define Objective functions
    mpc.obj = Objective(expr = sum(mpc.weight[t]*(mpc.deltaSetPointPos[t]+mpc.deltaSetPointNeg[t])+ mpc.cCost[t]*mpc.controlLevel[t] +mpc.dCost[t]*(mpc.controlLevelPos[t]+mpc.controlLevelNeg[t]) for t in mpc.time), sense = minimize )

    # Define Constraints
    # constraint calculating absolute value for change in Control level
    def controlLevelNegcnstr(mpc, t):
        if t == 0:
            return Constraint.Skip
        else:
            constr = mpc.controlLevel[t]-mpc.controlLevel[t-1]
            return constr == mpc.controlLevelPos[t] - mpc.controlLevelNeg[t]
    mpc.controlLevelNegcnstr = Constraint( mpc.time, rule= controlLevelNegcnstr)

    # constraint calculating absolute value for change in SoC
    def deltaSetPointcnstr(mpc, t):
        constr = mpc.SoC[t]-mpc.setPoint[t]
        return constr == mpc.deltaSetPointPos[t] - mpc.deltaSetPointNeg[t]
    mpc.deltaSetPointcnstr = Constraint( mpc.time, rule= deltaSetPointcnstr)
    
    # constraint constraints the controller level from changing more than one level up in an hour
    def constrDMax(mpc, t):
        if t == 0:
            return Constraint.Skip
        else:
            Constrnt = (mpc.controlLevel[t]-mpc.controlLevel[t-1] <= mpc.dMax)
            return Constrnt
    mpc.constrDMax = Constraint( mpc.time, rule= constrDMax )
    
    # constraint constraints the controller level from changing more than one level down in an hour
    def constrDMin(mpc, t):
        if t == 0:
            return Constraint.Skip
        else:
            Constrnt = (mpc.controlLevel[t]-mpc.controlLevel[t-1] >= -1*mpc.dMax)
            return Constrnt
    mpc.constrDMin = Constraint( mpc.time, rule= constrDMin )    
    
    # constraint initializes the first value of the Control level
    mpc.controlSoCInicnstr = Constraint(expr =  mpc.controlLevel[0] == mpc.controlLevelIni)

    # constraint initializes the first value of the SoC
    def constrSoCini(mpc,t): 
        if t == 0:
            return mpc.SoC[t] == mpc.SoCIni
        else:
            return Constraint.Skip
    mpc.constrSoCini = Constraint( mpc.time, rule= constrSoCini)
    
    # constraint calculates the next value of the SoC
    def constrSoC(mpc,t): 
        if t == 0:
            return Constraint.Skip
        else:
            Constrnt = (mpc.SoC[t] == mpc.SoC[t-1] + 139.85*mpc.controlLevel[t-1]/722 + mpc.SoCDiff[t-1])
            return Constrnt
    mpc.constrSoC = Constraint( mpc.time, rule= constrSoC)
    
    # Return the model
    return mpc


