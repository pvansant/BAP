import numpy as np
import functions as fs

def normalizeData(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

X, yWind = fs.retrieveWindData() # TODO new version

yWind = normalizeData(yWind) # normalize the data
peakWindPower = 12000 # [W] # peak power of wind generation
yWind = peakWindPower*yWind # [Wh/hour] # scale the data

X, ySolar = fs.retrieveSolarData()

ySolar = normalizeData(ySolar) # normalize the data
peakSunPower = 16470 # [W] # peak power of solar generation
ySolar = peakSunPower*ySolar # [Wh/hour] # scale the data

X, yDemand = fs.retrieveDemandData()

np.savez('data_original_V2_with-scaling', sunOutput=ySolar, windOutput=yWind, demandOutput=yDemand)