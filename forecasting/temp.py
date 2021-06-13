import numpy as np
# import functions as fs

def normalizeData(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def altNormalizeData(x, max, min):
    return (x-min)/(max-min)


realSolar = np.load('unscaledData/realSolar.npy')
predSolar = np.load('unscaledData/predSolar.npy')
realWind = np.load('unscaledData/realWind.npy')
predWind = np.load('unscaledData/predWind.npy')
realDemand = np.load('unscaledData/realDemand.npy')
predDemand = np.load('unscaledData/predDemand.npy')

# 31-12-12 23:42,0.985 # 105194
# 31-12-13 22:42,0.726 # 113953

print('realWind shape before slicing', realWind.shape)
realWind = realWind[105194-2:113953+1-2]
predWind = predWind[105194-2:113953+1-2]
print('realWind shape after slicing', realWind.shape)


# print('should be 0.985:',realWind[0])
# print('should be 0.726:',realWind[-1])

# peakWindPower = 16000 # [W] # peak power of wind generation
# peakSunPower = 13680 # [W] # peak power of solar generation
# averDemandPower = 41000 # [W] # peak power of demand


# ### sun scaling
# realSolar = normalizeData(realSolar) # normalize the data
# realSolar = peakSunPower*realSolar # [Wh/hour] # scale the data

# predSolar = normalizeData(predSolar) # normalize the data
# # predSolar = altNormalizeData(predSolar, np.max(realSolar), np.min(realSolar)) # normalize the data
# predSolar = peakSunPower*predSolar # [Wh/hour] # scale the data

# ### wind scaling
# realWind = normalizeData(realWind) # normalize the data
# realWind = peakWindPower*realWind # [Wh/hour] # scale the data

# predWind = normalizeData(predWind) # normalize the data
# # predWind = altNormalizeData(predWind, np.max(realWind), np.min(realWind)) # normalize the data
# predWind = peakWindPower*predWind # [Wh/hour] # scale the data

# # ### demand scaling
# # realDemand = normalizeData(realDemand) # normalize the data
# # realDemand = averDemandPower*realDemand # [Wh/hour] # scale the data

# # predDemand = altNormalizeData(predDemand, np.max(realDemand), np.min(realDemand)) # normalize the data
# # predDemand = averDemandPower*predDemand # [Wh/hour] # scale the data


print('total generated in a year by realSolar:', np.sum(realSolar))
print('total generated in a year by predSolar:', np.sum(predSolar))
print('total generated in a year by realWind:', np.sum(realWind))
print('total generated in a year by predWind:', np.sum(predWind))
print('total generated in a year by realDemand:', np.sum(realDemand))
print('total generated in a year by predDemand:', np.sum(predDemand))


np.savez('dataForControl', 
realSolar = realSolar,
predSolar = predSolar,
realWind = realWind,
predWind = predWind,
realDemand = realDemand,
predDemand = predDemand)


# dataForControl = np.load('dataForControl.npz')
# realSolar =     dataForControl['realSolar']
# predSolar =     dataForControl['predSolar']
# realWind =      dataForControl['realWind']
# predWind =      dataForControl['predWind']
# realDemand =    dataForControl['realDemand']
# predDemand =    dataForControl['predDemand']