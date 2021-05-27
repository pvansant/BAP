print('\n\nrunning...\n')
# import functions as fs
import numpy as np


# date, id, charger on/off, charge state, pv voltage, pv current, ...

data = np.genfromtxt('temp.csv', dtype=None, encoding=None, delimiter=',', skip_header=0)
 
hour = []
for i in data['f0']: hour.append(int(i[11:13])) 
hour = np.array(hour)

''' 
# this piece of code can be used to make every entry corresponding to a single hour unique 
# instead of only ranging from 0 to 23 every day
hour = []
for i in data['f0']: hour.append(int(i[5:7]+i[8:10]+i[11:13])) 
hour = np.array(hour)
'''

data = np.concatenate((hour[:,np.newaxis],data['f4'][:,np.newaxis],data['f5'][:,np.newaxis]), axis=1, dtype=None)


# calculate power from the voltage and current
newData = []
for i in data: newData.append(i[1]*i[2])

# mean for every hour
newNewData = []
w = 1
for i in range(0,len(data)):
    if data[i,0] == data[i-1,0] and i > 0: # equal; calculate weighted mean
        newNewData[-1] = (newNewData[-1]*w+newData[i])/(w+1)
        w += 1
    else: # not equal; reset weight     
        newNewData.append(newData[i])
        w = 1


print('\n...finished\n\n')