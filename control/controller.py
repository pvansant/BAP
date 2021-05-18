'''
Authors: Aart Rozendaal and Pieter Van Santvliet
Description: In this script, the functionality of the controller is programmed
It used different functions and a loop to control the system.
'''

import os

from numpy.core.function_base import linspace
os.system('cls') # clears the command window
import datetime as dt; start_time = dt.datetime.now()
# display a "Run started" message
print('Run started at ', start_time.strftime("%X"), '\n')


#optimizer setup

#loop
    # Retrieve the measurements (the real data set of k=0)

    # Make predictions using the ANN for demand and generation (for k = 1 till k = 169 (7 days))

    # Solve the equations using the GEKKO optimizer

    # Implement the first control strategy

    # Move one time step further and loop back

    # Use the wait command (only for implementation)


# print the runtime
print('\nRuntime was', (dt.datetime.now() - start_time).total_seconds(), 'seconds')




