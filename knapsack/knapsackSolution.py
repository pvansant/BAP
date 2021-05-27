"""
Building your fisrt Mathematical Programming Model in Pyomo
by Pedro P. Vergara, PhD. TU Eindhoven, 2020
"""
"""
        Tutorial: Implementing your first mathematical programming model in Pyomo. See the PDF in this folder for the mathematical formulation.
        
        Problem description: Amont a set of tools (A) with benefit (b) and weight (w), selects the maximum number of tools that I can carry
                             and will give me the greatest benefit knowing that I can carry a maximum total weight of Wmax.
            
        Inputs: A : Sets of items available for purchase    Type: Set
                b : benefit of each item                    Type: Parameter
                w : weight of each item                     Type: Parameter
                Wmax : max weight that can be carried       Type: Parameter
        
        Outputs: Which outpus to carry                      Type: Decision variable (binary, can be 0 or 1)
"""

"""
This Pythion script is composed of four setps, described as follows:
    Step I:   Upload all the python packages required. Upload Pyomo.
    Step II:  Define the input parameters. Create the required dictionaries for Pyomo. 
    Step III: Build the mathematical model. Check the file knapsack_problem_model.py for more information.
    Step IV:  Solve the model. Define the solver and print some results. 

"""
# -------------------------------------------------------------------
# ------------------------    Part I     ----------------------------
# -------------------------------------------------------------------

from pyomo.environ import *
import knapsackProblem as kn

# -------------------------------------------------------------------
# ------------------------    Part II     ---------------------------
# -------------------------------------------------------------------

# Upload the input data. Check the function python file and the attached PDF for an definition of these inputs sets and parameters.
A = [1, 22, 33, 4]
b_ini = [8, 3, 6, 11]
w_ini = [5, 7, 4, 3]
Wmax = 14

# Note: I structured this first part as it is show here, as it might be the case that (for your application) this input
#       data might comes from a pre-processing stage. Once you have your input data ready, we can create the dictionaries
#       for Pyomo.

# Create the dictionaries to pass the models data to Pyomo.
b = {A[i]: b_ini[i] for i in range(len(A))}
w= {A[i]: w_ini[i] for i in range(len(A))}

# -------------------------------------------------------------------
# ------------------------  Part III     ----------------------------
# -------------------------------------------------------------------

# Create the model calling the function that containts the mathematical programming model.
model =  kn.knapsack_problem_model(A,b,w,Wmax)

# -------------------------------------------------------------------
# ------------------------  Part IV     ----------------------------
# -------------------------------------------------------------------
# Define the solver to solve the model. 
solver = SolverFactory('glpk') 

# Note: You need to install GLPK in your Python invironment before executing this line. 

# Solve the model.
solver.solve(model)

# Note:  GLPK is a mixed-integer linear programming solver. There are plenty of solvers available. Some of them are open-source
#        and others not. To name a few: CPLEX, IPOPT, BONMIN, GUROBI, GLPK, etc.

# Print some results
print('Solving the Classic Knapsack Problem using Pyomo')
print('Printing results...Wait')

print('Item\tSelected (Yes = 1, No = 0)')
for i in model.A:
	print('%d\t%d'%(i,model.x[i].value))
print('Total benefit: %0.2f'%(sum(model.b[i].value*model.x[i].value for i in model.A)))
print('Total weight: %0.2f'%(sum(model.w[i].value*model.x[i].value for i in model.A)))
print('Max weight: %0.2f'%(model.Wmax.value))

# Note: Once you have undestand this code. I recommend to you to solve it for A = [1,22,33,4].
#       This will help you to understand the concept of 'Set'. 

# Good look with your work. 
    