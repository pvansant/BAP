'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the variables for optimization are updated,
the time horizon is updated and the new optimization is calculated
'''

import os

from numpy.core.function_base import linspace
os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

# Importing libraries
from gekko import GEKKO
import matplotlib.pyplot as plt
import numpy as np
from random import random


# Solve the model 

mpc.solve(disp=False, remote=False)     # Solve Locally


# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')