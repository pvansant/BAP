'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, three functions are writen regarding reading prediction data, reading the current measurement and calculating the influence of the controller using the random library and the current hour.
'''
# add libraries
import numpy as np
import random as rnd

'''
Function: calculates the predicted change in SoC due to the predictions
inputs: index - current hour
length - length of the time horizon
sun - predictions of solar generation over the time horizon
wind - predictions of wind generation over the time horizon
demand - predictions of demand over the time horizon
output: SoCChange - predicted change in SoC of the battery due to the predicted generation and demand
'''
def readPredictions(index,length,sun,wind,demand):
    SoCChange = [] # initiate list
    length = length +index
    for i in range(index,length):
        SoCChange.append((sun[i] + wind[i] - demand[i])/772.0) # read the predictions for every hour and scale Wh to SoC
    return SoCChange # return Change in SoC

'''
Function: Reads the current value of the SoC
inputs: index - current hour
sun - measured solar generation at the current hour
wind - measured wind generation at the current hour
demand - measured demand at the current hour
output: SoCChange - actual change in SoC of the battery due to the measured generation and demand
'''
def readMeasurement(index,sun,wind,demand): 
    SoCChange = (sun[index] + wind[index] - demand[index])/772.0 # read the measured generation and demand and scale Wh to SoC
    return SoCChange # return Change in SoC

'''
Function: calculates the effect of the controller at the current hour using the control level
inputs: index - current hour
controlLevel - current control level
output: SoCChange - actual change in SoC of the battery due to the control level
'''
def determineControlSoC(index, controlLevel): # TODO switch the first two
    ControlSoC = 0 # intialize the change in SoC as zero 
    hour = index%24 # calculate which hour of the day it is
    if controlLevel > 0: # control level 1 or higher
        if hour > 5 and hour < 21: 
            for i in range(12): # calculate at random who is not home
                temp = rnd.randint(1,1000)
            if temp < 408:
                ControlSoC += 70 # when not home boiler is turned off
            else:
                ControlSoC += 0
        if controlLevel > 1: # control level 2 or higher
            if hour == 20 or hour == 21 or hour == 4 or hour == 5:
                ControlSoC += 60 # use lighting at 50% during these hours
            elif hour > 21 or hour < 4:
                ControlSoC += 120 # dont use lighting during these hours
            else:
                ControlSoC += 0 # no savings during the day
            if controlLevel > 2:# control level 3 or higher
                if hour > 5 and hour < 22:
                    for i in range(12): # calculate at random who is using the wachingmachine
                        temp = rnd.randint(1,1000)
                        if temp < 400:
                            ControlSoC += 10.5 # when using it it can only use eco mode
                else:
                    ControlSoC += 0
                if controlLevel >3:# control level 4
                    if hour > 17 and hour < 22:
                        for i in range(12):# calculate at random who is using the stove
                            temp = rnd.randint(1,1000)
                            if temp < 250: 
                                ControlSoC += 281.25# when using it it can only use one on high and one on medium
    
    ControlSoC = ControlSoC/722 # convert Wh to SoC
                                                  
    return ControlSoC # return the change in SoC