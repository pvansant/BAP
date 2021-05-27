print('\n\nrunning...\n')
# import functions as fs
import numpy as np


# date, id, charger on/off, charge state, pv voltage, pv current, ...
data = np.genfromtxt('forecasting\generationData\2019Bikechargerdata_voltageandcurrentWithoutQuotes.csv', dtype=None, encoding=None, delimiter=',', skip_header=0)
data = np.genfromtxt('forecasting\generationData\temp.csv', dtype=None, encoding=None, delimiter=',', skip_header=0)
 
# extract the hour from the data
hour = []
for i in data['f0']: hour.append(int(i[9:11])) 
hour = np.array(hour)

# the piece of code below can be used to make every entry corresponding to a single hour unique 
# instead of only ranging from 0 to 23 every day
# hour = []
# for i in data['f0']: hour.append(int(i[5:7]+i[8:10]+i[11:13])) 
# hour = np.array(hour)

# create a numpy array from the hour, volt, and current data
data = np.concatenate((hour[:,np.newaxis],data['f4'][:,np.newaxis],data['f5'][:,np.newaxis]), axis=1, dtype=None)


# calculate power from the voltage and current
irregularPowerData = []
for i in data: irregularPowerData.append(i[1]*i[2])

# mean for every hour
hourlyPowerData = []
w = 1 # weight factor for calculating the mean
print('\n\nforloop starts\n\n')
for i in range(0,len(data)):
    # first iteration; initialization
    if i == 0: hourlyPowerData.append(irregularPowerData[i])

    # if an hour is repeated, calculate the mean for that hour
    elif data[i,0] == data[i-1,0]: 
        hourlyPowerData[-1] = (hourlyPowerData[-1]*w+irregularPowerData[i])/(w+1) # weight is equal for every variable
        w += 1 # this requires the weight factor to be increased throughout iterations

    # if a new hour is detected, it has to be initialized and added to the final data array
    elif data[i,0]-data[i-1,0] == 1 or data[i-1,0]-data[i,0] == 23:  
        hourlyPowerData.append(irregularPowerData[i])
        w = 1 # weight factor must be reset

    # if something goes wrong # FIXME missing hours
    else:
        print('hour missing at i is',i)
        
# save the acquired array
# np.save('processedSolarData', hourlyPowerData)

print('\n...finished\n\n')