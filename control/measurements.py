import numpy as np



def readMeasurement(index,length,sun,wind,demand):
    SoCChange = np.array() 
    length = length +index
    for i in range(index,length):
        SoCChange[i] = (sun[i] + wind[i] - demand[i])*772 
    return SoCChange
