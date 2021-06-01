import numpy as np



def readMeasurements(index,length,sun,wind,demand):
    SoCChange = []
    length = length +index
    for i in range(index,length):
        SoCChange.append((sun[i] + wind[i]*0.5 - demand[i]*1.3)/772.0)
        #SoCChange.append((sun[i] + wind[i] - demand[i])/772.0)
    return SoCChange
