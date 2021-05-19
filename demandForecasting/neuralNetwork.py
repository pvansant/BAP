'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the time and temperature at a certain moment 
are used to predict the demand of a certain group of houses. This is done 
using an Artificial Neural Network.
'''

import os
os.system('cls') # clears the command window
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

import functions as f
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import sklearn.impute
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# extracting input data from txt file
try:
    data = np.genfromtxt('demandForecasting/relevantData/x_2013-data-points.txt', 
    dtype=float, delimiter=',', skip_header=33)
    X = data[:,[1,2,7,10,21,23]] # only relevant stuff:
    # select YYYYMMDD (col 1; datum), HH (col 2; hour), T (col 7; temperature), 
    # SQ (col 10; sunshine duration), R (col 21; rain), O (col 23; storm)
except:
    print('Error while retrieving input data'); exit()

# we want the weeknumber and daynumber instead of the date
timeInfo = np.empty((0,2), int)
for i in X[:,0]:
    # get the date info from the data file
    year = int(str(i)[0:4])
    month = int(str(i)[4:6])
    day = int(str(i)[6:8])
    # make a date from the date info
    time = dt.datetime(year, month, day)
    # timeInfo will contain the weeknumber (%V) and the daynumber (%u)
    timeInfo = np.append(timeInfo, np.array([[time.strftime("%V"),time.strftime("%u")]]), axis=0)

# the date-column is replaced by a weeknumber and daynumber column
X = np.append(timeInfo, np.delete(X,0,1), 1)

# print(X[0:100,:])

# extracting output data from csv file
try:
    data = np.genfromtxt('demandForecasting/relevantData/Zonnedael - slimme meter dataset - 2013 - Levering.csv', 
    dtype=float, delimiter=',', skip_header=1, skip_footer=34992)
    y = data[:,2:-2] # only relevant stuff:
    # select YYYYMMDD (col 1; datum), HH (col 2; hour), T (col 7; temperature), 
    # SQ (col 10; sunshine duration), R (col 21; rain), O (col 23; storm)
except:
    print('Error while retrieving output data'); exit()

y = y.reshape(-1,4,y.shape[-1]).sum(1)

# from https://stackoverflow.com/q/18689235 # FIXME using pandas
y = np.where(np.isnan(y), ma.array(y, mask=np.isnan(y)).mean(axis=1)[:, np.newaxis], y)
y = y.sum(1)

####################################### COPY OF WEATHER NN BELOW

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
# f.printSets(X_train, X_test, y_train, y_test) # enable to print set shapes

# checking and handling missing values # FIXME improve NaN handling
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

# from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# baseline_model().summary() # enable to print a summary of the NN model

model = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=100, verbose=2)
# verbose =0 will show nothing; =1 will show animated progress; =2 will mention the number of epochs

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', model))
pipeline = Pipeline(estimators)

### make a single model and evaluate it with the test set
'''
pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)
mse_krr = mean_squared_error(y_test, y_pred)

print('\nThe MSE is', mse_krr)
print('The RMSE is', mse_krr**0.5)
print('The relative error is', mse_krr**0.5/30800)
'''

### make a single model (without the pipeline) and show the learning curve
'''
history = model.fit(X_train,y_train)
# print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
'''

### make multiple models using cross_val_score and evaluate it using validation sets from the training set
'''
kfold = KFold(n_splits=3)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold)

print("Standardized: %.7f (%.7f) MSE" % (results.mean(), results.std()))
'''

# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')
