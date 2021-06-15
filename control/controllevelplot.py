import matplotlib.pyplot as plt
controlLevel = [0,1,2,3,4]
saving = [0,253,283,301.9,442.5]
controlLevellin = [0,116.49,116.49*2,116.49*3,116.49*4]

plt.plot(controlLevel,controlLevellin, c = '#0C7CBA', ls = '-')
plt.plot(controlLevel,saving, c = '#0C7CBA', ls = '',marker ='o') 
plt.xlabel("Control level")
plt.ylabel("Estimated savings [Wh]")
plt.show()