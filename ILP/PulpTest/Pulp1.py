from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, LpSolverDefault

# Define the problem
problem = LpProblem("Simple_Neural_Network", LpMinimize)

# Variables
w = LpVariable("w", lowBound=-10, upBound=10)  # Weight
b = LpVariable("b", lowBound=-10, upBound=10)  # Bias
output = LpVariable("output", lowBound=0)      # ReLU output
z = LpVariable("z", cat=LpBinary)              # Binary variable for ReLU activation

# Input and target output (can be changed)
x = 2  # Input sample
y = 5  # Target output

# ReLU constraints
M = 100  # Large constant for big-M method
problem += output >= w * x + b  # Output should be at least wx + b
problem += output >= 0          # ReLU cannot be negative
problem += output <= w * x + b + M * (1 - z)  # Big-M constraint

# Loss function (absolute error)
loss = LpVariable("loss", lowBound=0)
problem += loss >= output - y
problem += loss >= y - output

# Objective: Minimize loss
problem += loss

# Solve the LP
problem.solve()

# Print results
print(f"Optimal w: {w.varValue}")
print(f"Optimal b: {b.varValue}")
print(f"Output: {output.varValue}")
print(f"Loss: {loss.varValue}")
