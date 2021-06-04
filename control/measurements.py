import numpy as np
import random as rnd


def readPredictions(index,length,sun,wind,demand):
    SoCChange = []
    length = length +index
    for i in range(index,length):
        SoCChange.append((sun[i] + wind[i]*0.5 - demand[i])/772.0)
        #SoCChange.append((sun[i] + wind[i] - demand[i])/772.0)
    return SoCChange

def readMeasurement(index,length,sun,wind,demand): #TODO 
    SoCChange = []
    length = length +index
    for i in range(index,length):
        SoCChange.append((sun[i] + wind[i]*0.5 - demand[i])/772.0)
        #SoCChange.append((sun[i] + wind[i] - demand[i])/772.0)
    return SoCChange

def determineControlSoC(index, controlLevel):
    ControlSoC = 0

    hour = index%24

    if controlLevel > 0:
        if hour == 20 or hour == 21 or hour == 4 or hour == 5:
            ControlSoC -= 60
        elif hour > 21 or hour < 4:
            ControlSoC -= 120
        else:
            ControlSoC += 0
        
        if controlLevel > 1:
            if hour > 5 and hour < 21:
                for i in range(12):
                    temp = rnd.randint(1,1000)
                    if temp < 408:
                        ControlSoC -= 70
            else:
                ControlSoC += 0
            if controlLevel > 2:
                if hour > 5 and hour < 22:
                    for i in range(12):
                        temp = rnd.randint(1,1000)
                        if temp < 400:
                            ControlSoC -= 10.5
                else:
                    ControlSoC += 0
                if controlLevel >3:
                    if hour > 17 and hour < 22:
                        for i in range(12):
                            temp = rnd.randint(1,1000)
                            if temp < 250:
                                ControlSoC -= 281.25
    
    ControlSoC = ControlSoC/722
                            
                        
    return ControlSoC