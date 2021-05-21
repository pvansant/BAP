'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the optimization for the mpc is run. 
'''

import os

from numpy.core.function_base import linspace
os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')

# Importing libraries
from pyomo.environ import *
from pyomo.dae import *


