'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, weather data is used to predict the average capacity factor of wind turbines in Rotterdam. This is done using an Artificial Neural Network.
'''

import os
os.system('cls') # clears the command window
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

import functions as f
import numpy as np
import sklearn as sk
import sklearn.impute
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# extracting data from csv file
try:
    data = np.genfromtxt('weatherForecasting/relevantData/Training-1_y.csv', 
    dtype=float, delimiter=',', skip_header=1, skip_footer=1)
    y = data[:,1] # only relevant stuff; all rows of column 1
except:
    print('Error while retrieving y'); exit()

# extracting data from txt file
try:
    data = np.genfromtxt('weatherForecasting/relevantData/Training-1_X.csv', 
    dtype=float, delimiter=',', skip_header=33)
    X = data[:,3:] # only relevant stuff; all rows of column 3 till end
except:
    print('Error while retrieving X')
    exit()

# print(X[:5,:2])
# print(y[:2])
# print(X[-5:,:2])
# print(y[-2:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
# f.printSets(X_train, X_test, y_train, y_test) # enable to print set shapes

# checking and handling missing values 
imp = sk.impute.SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

# from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

epochs = 100
batch_size = 200
verbose = 2         # 0 to show nothing; 1 or 2 to show the progress
n_splits = 3

model = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=verbose)

# evaluate model with standardized dataset
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
kfold = KFold(n_splits=n_splits)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold)

# print("\nStandardized: %.7f (%.7f) MSE" % (results.mean(), results.std()))
RRMSE = abs(results.mean().item())**0.5
'''

### print the results
'''
print('\n\n')
baseline_model().summary() # enable to print a summary of the NN model
print('\nParameters:')
print('\tepochs:', epochs)
print('\tbatch_size:', batch_size)
print('\tn_splits:', n_splits)
print('\tX_train shape:', X_train.shape)
print('\nRelative Root MSE becomes: {:.1%}'.format(RRMSE))
'''

# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')