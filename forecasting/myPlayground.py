print('\n\nrunning...\n')

import numpy as np
import datetime as dt
import math



data = np.load('tempData.npy')

# data = np.genfromtxt('forecasting/generationData/2019Bikechargerdata_voltageandcurrentWithoutQuotes.csv', dtype=None, encoding=None, delimiter=',', skip_header=0, comments='#')
# note on the data: date, id, charger on/off, charge state, pv voltage, pv current, ...
# np.save('tempData', arr=data) # saving the data (to decrease runtime)


# data = np.genfromtxt('forecasting/generationData/temp.csv', dtype=None, encoding=None, delimiter=',', skip_header=0)
 

# return the hour of the year, given a certain date
def hourOfYear(date): 
    beginningOfYear = dt.datetime(date.year, 1, 1, tzinfo=date.tzinfo)
    return int((date - beginningOfYear).total_seconds() // 3600)


# extract the hour from the data
hour = []
for t in data['f0']: hour.append(int(t[9:11])) 
hour = np.array(hour)


# extract the date from the data
time = []
for t in data: 
    time.append(dt.datetime(year=2019, month=int(t['f0'][3:5]), day=int(t['f0'][0:2]), hour=int(t['f0'][9:11]))) 
time = np.array(time)


# convert the date to the hour of the year
timeAsHour = []
for t in time: timeAsHour.append(hourOfYear(t))
timeAsHour = np.array(timeAsHour)


# the piece of code below can be used to make every entry corresponding to a single hour unique 
# instead of only ranging from 0 to 23 every day
# hour = []
# for i in data['f0']: hour.append(int(i[5:7]+i[8:10]+i[11:13])) 
# hour = np.array(hour)

# create a numpy array from the hour, volt, and current data
data = np.concatenate((hour[:,np.newaxis],data['f4'][:,np.newaxis],data['f5'][:,np.newaxis]), axis=1, dtype=None)


# calculate power from the voltage and current
irregularPowerData = []
for t in data: irregularPowerData.append(t[1]*t[2])
irregularPowerData = np.array(irregularPowerData)


# initialize
hourlyPowerData = []
w = 1 # weight factor for calculating the mean

# mean for every hour
for i in range(0,len(timeAsHour)):
    # first iteration; initialization
    if i == 0: hourlyPowerData.append(irregularPowerData[i])

    # if an hour is repeated, calculate the mean for that hour
    elif timeAsHour[i]-timeAsHour[i-1] == 0: 
        hourlyPowerData[-1] = (hourlyPowerData[-1]*w+irregularPowerData[i])/(w+1) # weight is equal for every variable
        w += 1 # this requires the weight factor to be increased throughout iterations

    # if a new hour is detected, it has to be initialized and added to the final data array
    elif timeAsHour[i]-timeAsHour[i-1] == 1:  
        hourlyPowerData.append(irregularPowerData[i])
        w = 1 # weight factor must be reset

    elif i == 127830:  
        hourlyPowerData.append(irregularPowerData[i])
        w = 1 # weight factor must be reset

    elif timeAsHour[i]-timeAsHour[i-1] < 0:
        # TODO incomplete
        w = 1 # weight factor must be reset
        continue

    # if something goes wrong 
    elif timeAsHour[i]-timeAsHour[i-1] > 1:
        deviation = timeAsHour[i]-timeAsHour[i-1]
        for k in range(int(deviation)): hourlyPowerData.append(np.nan)
        w = 1 # weight factor must be reset

    else:
        print('unkown error at i =',i)

# convert list to numpy array
hourlyPowerData = np.array(hourlyPowerData)

############################################# ALTERNATIVE METHOD

# initialize
alternativeMethod = np.empty(np.max(timeAsHour)+1)
alternativeMethod[:] = np.NaN
i = 0
w = np.ones(np.max(timeAsHour)+1)


# mean for every hour
for t in timeAsHour:

    if math.isnan(alternativeMethod[t]):
        alternativeMethod[t] = irregularPowerData[i]
    else:
        alternativeMethod[t] = (alternativeMethod[t]*w[t]+irregularPowerData[i])/(w[t]+1) # weight is equal for every variable
        w[t] += 1 # this requires the weight factor to be increased throughout iterations

    i += 1


# save the acquired array
# np.save('processedSolarData', hourlyPowerData)

print('\n...finished\n\n')