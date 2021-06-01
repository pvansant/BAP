
'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the optimization for the mpc is initialized. 
Constants and variables are defined and a model is created.
'''

import os

# Importing libraries
from pyomo.environ import *
from pyomo.dae import *

'''
Inputs:
SoC                     - integer respresenting SoC
SoCDiff                 - Set of length 169 with the predicted difference in SoC for the coming hour
setPoint                - Set of length 169 with the Setpoint for every hour
weight                  - Set of length 169 with the weight of the MSE for every hour
dCost                   - Set of length 169 with the weigth penalty for changing the control
dMax                    - integer respresenting the maximum change in the control signal per hour
controlLevelIni         - integer respresenting current control level

output:
Model
'''


def modelPredictiveControl(time,SoCIni,SoCDiff,setPoint,weight,dCost,dMax,controlLevelIni):

    #initializing the mpc model using pyomo
    mpc = ConcreteModel()

    # Define the Set
    mpc.time = Set(initialize= time) # Define the time steps from 0 to 168 hours
   
    # Define Parameters   
    mpc.SoCDiff = Param(mpc.time, initialize=SoCDiff, mutable=True)
    mpc.setPoint = Param(mpc.time, initialize=setPoint, mutable=True)
    mpc.weight = Param(mpc.time, initialize=weight, mutable=True)
  #  mpc.dCost = Param(mpc.time, initialize=dCost, mutable=True )
    mpc.SoCIni = Param(initialize=SoCIni, mutable = True)
    mpc.controlLevelIni = Param(initialize=controlLevelIni, mutable = True)
    mpc.dMax = Param(initialize=dMax, mutable = True)
    
    # Define Variables
    mpc.controlLevel = Var(mpc.time, within = Integers, bounds = (0,3))
    mpc.SoC = Var(mpc.time, within = Reals)
    mpc.deltaSetPointPos = Var(mpc.time , within = NonNegativeReals)
    mpc.deltaSetPointNeg = Var(mpc.time , within = NonNegativeReals)
  #  mpc.controlLevelPos = Var(mpc.time , within = NonNegativeIntegers)
  #  mpc.controlLevelNeg = Var(mpc.time , within = NonPositiveIntegers)

    # Define Objective functions
    #mpc.obj = Objective(expr = sum(mpc.weight[i]*(mpc.setPoint[i]-mpc.SoC[i]) for i in mpc.time), sense = minimize )
    mpc.obj = Objective(expr = sum(mpc.weight[i]*(mpc.deltaSetPointPos[i]+mpc.deltaSetPointNeg[i]) for i in mpc.time), sense = minimize )
    #mpc.obj = Objective(expr = sum(mpc.weight[i]*(mpc.SoC[i]-mpc.setPoint[i]) for i in mpc.IDX), sense = minimize )
    #+ mpc.dCost[i]*(mpc.controlLevelPos[i]-mpc.controlLevelNeg[i])
    # Define Constraints
    
    '''
    def controlLevelPoscnstr(mpc, i):
        constr = mpc.controlLevel[i]-mpc.controlLevel[i-1]
        return constr == mpc.controlLevelPos[i]
    mpc.controlLevelPoscnstr = Constraint( mpc.IDX, rule= controlLevelPoscnstr)

    def controlLevelNegcnstr(mpc, i):
        constr = mpc.controlLevel[i]-mpc.controlLevel[i-1]
        return constr == mpc.controlLevelNeg[i]
    mpc.controlLevelNegcnstr = Constraint( mpc.IDX, rule= controlLevelNegcnstr)
    

    def deltaSetPointPoscnstr(mpc, i):
        constr = mpc.SoC[i]-mpc.setPoint[i]
        return constr == mpc.deltaSetPointPos[i]
    mpc.deltaSetPointPoscnstr = Constraint( mpc.time, rule= deltaSetPointPoscnstr)
    '''
    def deltaSetPointcnstr(mpc, t):
        constr = mpc.SoC[t]-mpc.setPoint[t]
        return constr == mpc.deltaSetPointPos[t] - mpc.deltaSetPointNeg[t]
    mpc.deltaSetPointcnstr = Constraint( mpc.time, rule= deltaSetPointcnstr)
    
    def constrDMax(mpc, t):
        if t == 0:
            return Constraint.Skip
        else:
            Constrnt = (mpc.controlLevel[t]-mpc.controlLevel[t-1] <= mpc.dMax)
            return Constrnt
    mpc.constrDMax = Constraint( mpc.time, rule= constrDMax )
    
    def constrDMin(mpc, t):
        if t == 0:
            return Constraint.Skip
        else:
            Constrnt = (mpc.controlLevel[t]-mpc.controlLevel[t-1] >= -1*mpc.dMax)
            return Constrnt
    mpc.constrDMin = Constraint( mpc.time, rule= constrDMin )    
    
    mpc.controlSoCInicnstr = Constraint(expr =  mpc.controlLevel[0] == mpc.controlLevelIni)

    def constrSoCini(mpc,t): 
        if t == 0:
            return mpc.SoC[t] == mpc.SoCIni
        else:
            return Constraint.Skip
    mpc.constrSoCini = Constraint( mpc.time, rule= constrSoCini)
    
    def constrSoC(mpc,t): 
        if t == 0:
            return Constraint.Skip
        else:
            Constrnt = (mpc.SoC[t] == mpc.SoC[t-1] + 139.85*mpc.controlLevel[t-1]/722 + mpc.SoCDiff[t-1] )
            return Constrnt
    mpc.constrSoC = Constraint( mpc.time, rule= constrSoC)
    
    # Return the model
    return mpc


