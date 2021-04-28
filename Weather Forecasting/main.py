import csv

with open('Weather Forecasting/Data/test.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

print(data)
print(type(data))

## simpler thing that does the same
# f = open("Weather Forecasting/Data/test.txt", "r")
# print(f.read())
# f.close()