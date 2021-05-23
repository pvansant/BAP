print('\n\nrunning...\n')
# import functions as fs
import numpy as np


# date, id, charger on/off, charge state, pv voltage, pv current, ...

data = np.genfromtxt('temp.csv', dtype=None, encoding=None, delimiter=',', skip_header=0)

hour = []
for i in data['f0']: hour.append(int(i[11:13])) 
hour = np.array(hour)

data = np.concatenate((hour[:,np.newaxis],data['f4'][:,np.newaxis],data['f5'][:,np.newaxis]), axis=1, dtype=None)

print(data)


print('\n...finished\n\n')