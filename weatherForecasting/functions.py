# Authors: Aart Rozendaal and Pieter Van Santvliet

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.impute
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def printSets(X_train, X_test, y_train, y_test):
    '''
    Print the shapes of the train and test sets.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : numpy arrays
        The sets that are to be printed.

    Output
    ------
    For every set, the shape is printed.
    '''
    print("shape of X_train: {}".format(X_train.shape))
    print("shape of X_test: {}".format(X_test.shape))
    print("shape of y_train: {}".format(y_train.shape))
    print("shape of y_test: {}".format(y_test.shape))


def retrieveGenerationData():
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
        print('Error while retrieving X'); exit()
    
    return X, y


### make a single model and evaluate it with the test set
def trainWithoutCurve(X_train, y_train, pipeline):
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_train)
    MSE = mean_squared_error(y_train, y_pred)
    rootMSE = MSE**0.5
    return MSE,rootMSE


### make a single model (without the pipeline) and show the learning curve
def trainWithCurve(X_train, y_train, model):
    history = model.fit(X_train,y_train)
    print(history.history.keys())
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


### make multiple models using cross_val_score and evaluate it using validation sets from the training set
def performCrossValidation(X_train, y_train, n_splits, pipeline):
    kfold = KFold(n_splits=n_splits)
    results = cross_val_score(pipeline, X_train, y_train, cv=kfold)

    MSE = results.mean().item()
    STD = results.std().item()
    rootMSE = abs(results.mean().item())**0.5

    return MSE,STD


### print the results
def printTrainingResults(X_train, epochs, batch_size, n_splits, baseline_model, MSE, rootMSE):
    print('\n\n')
    baseline_model().summary() # enable to print a summary of the NN model
    
    print('\nParameters:')
    print('\tepochs:\t\t', epochs)
    print('\tbatch_size:\t', batch_size)
    print('\tn_splits:\t', n_splits)
    print('\tX_train shape:\t', X_train.shape)
    
    print('\nMSE becomes: {:.4f}'.format(abs(MSE)))
    print('Root MSE becomes: {:.4f}'.format(rootMSE))