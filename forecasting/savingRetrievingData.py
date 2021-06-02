import numpy as np


##################################################################
###################### RETRIEVING DATA FILE ######################
##################################################################


# data = np.load('data_V1.npz')
# predSun = data['predSun']
# predWind = data['predWind']
# predDemand = data['predDemand']

# print('\nfile consists of:\n', data.files, '\n')
# print('shape sun:\t',predSun.shape)
# print('shape wind:\t',predWind.shape)
# print('shape demand:\t',predDemand.shape)


###################################################################
################### CREATING & SAVING DATA FILE ###################
###################################################################


# select data
predDemand = np.load('predictedDemand_V1.npy')
predWind = np.load('predictedWind_V1.npy')
weather = np.load('weatherData_V1.npy')

# to retrieve the original (for this the functions module is needed)
# import functions as fs
# weather = fs.retrieveWeatherData()
# np.save('weatherData_V1', weather)

def normalizeData(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

# selecting and scaling wind data
predWind = predWind[105192:113951+1] # select 2013
predWind = normalizeData(predWind) # normalize the data
peakWindPower = 12000 # [W] # peak power of wind generation
predWind = peakWindPower*predWind # [Wh/hour] # scale the data

# selecting and scaling solar data
predSun = weather[:,11] # [J/cm2] # select Global radiation
predSun = normalizeData(predSun) # normalize the data
peakSunPower = 16470 # [W] # peak power of solar generation
predSun = peakSunPower*predSun # [Wh/hour] # scale the data

np.savez('data_V2.npz', predSun=predSun, predWind=predWind, predDemand=predDemand)