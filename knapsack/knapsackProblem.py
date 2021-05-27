from pyomo.environ import *
"""
# Note: In Pyomo there are two ways for coding mathematical programming models: ConcreteModel() and AbstractModel().
        I prefer to use the ConcreteModel() as it gives us more flexibility. The model is coded as a function (as you can see below) 
        in which we can define the input parameters usind dictionariyes and other data-structures from Python. Thus, we can call the model
        everytime is required updating just the input information.
        
        In this example, we will implement the classic Knapsack Problem.
        
        Problem description: Amont a set of tools (A) with benefit (b) and weight (w), selects the maximum number of tools that I can carry
                             and will give me the greatest benefit knowing that I can carry a maximum total weight of Wmax.
            
        Inputs: A : Sets of items available for purchase    Type: Set
                b : benefit of each item                    Type: Parameter
                w : weight of each item                     Type: Parameter
                Wmax : max weight that can be carried       Type: Parameter
        
        Outputs: Which outpus to carry                      Type: Decision variable (binary, can be 0 or 1)
"""

def knapsack_problem_model(A,b,w,Wmax):
    # Define the type of Model
    model = ConcreteModel()

    # Note: the feature 'mutable' is set to 'True' for the Sets and Parameters in order
    #       to let Pyomo knows that these are values that can change every time the model is called.

    # Define the Sets 
    model.A = Set(initialize=A)
    
    # Define the Parameters
    model.b = Param(model.A, initialize=b, mutable=True)
    model.w = Param(model.A, initialize=w, mutable=True)
    model.Wmax = Param(initialize=Wmax, mutable=True)
    
    # Define the Variables
    model.x = Var(model.A, within=Binary)

    # Define the Objective Function
    model.value = Objective(expr =  sum(model.b[i]*model.x[i] for i in model.A), sense=maximize)
    
    # Define the Constraints
    model.weight = Constraint(expr =  sum(model.w[i]*model.x[i] for i in model.A) <= model.Wmax)
    
    # return the mathematical model
    return model