# Authors: Aart Rozendaal and Pieter Van Santvliet

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