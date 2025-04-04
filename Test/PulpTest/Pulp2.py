from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, value

# Define the LP problem
problem = LpProblem("Neural_Network_LP", LpMinimize)

# Input data (1 sample with 2 features)
X = [2, 3]  # Example input
y = 1       # Target output

# Define weight and bias variables
W1 = LpVariable("W1", -10, 10)
W2 = LpVariable("W2", -10, 10)
W3 = LpVariable("W3", -10, 10)
W4 = LpVariable("W4", -10, 10)
b1 = LpVariable("b1", -10, 10)
b2 = LpVariable("b2", -10, 10)

W5 = LpVariable("W5", -10, 10)
W6 = LpVariable("W6", -10, 10)
b3 = LpVariable("b3", -10, 10)

# Define ReLU outputs
Z1 = LpVariable("Z1", lowBound=0)  # ReLU applied on first hidden node
Z2 = LpVariable("Z2", lowBound=0)  # ReLU applied on second hidden node
Z3 = LpVariable("Z3")  # Output node
y_pred = LpVariable("y_pred", 0, 1)  # Binary output

# Big-M constraint for ReLU
M = 100  
Z1_bin = LpVariable("Z1_bin", cat=LpBinary)
Z2_bin = LpVariable("Z2_bin", cat=LpBinary)

# Hidden Layer Computation
problem += Z1 >= W1 * X[0] + W2 * X[1] + b1  # Z1 >= linear sum
problem += Z1 >= 0  # ReLU constraint
problem += Z1 <= W1 * X[0] + W2 * X[1] + b1 + M * (1 - Z1_bin)

problem += Z2 >= W3 * X[0] + W4 * X[1] + b2
problem += Z2 >= 0
problem += Z2 <= W3 * X[0] + W4 * X[1] + b2 + M * (1 - Z2_bin)

# Define bounds for McCormick constraints
Z1_LB, Z1_UB = 0, 10
Z2_LB, Z2_UB = 0, 10
W5_LB, W5_UB = -10, 10
W6_LB, W6_UB = -10, 10

# Define auxiliary variables
aux_W5_Z1 = LpVariable("aux_W5_Z1", lowBound=-100, upBound=100)
aux_W6_Z2 = LpVariable("aux_W6_Z2", lowBound=-100, upBound=100)

# McCormick Constraints for aux_W5_Z1 = W5 * Z1
problem += aux_W5_Z1 >= W5_LB * Z1 + W5 * Z1_LB - W5_LB * Z1_LB
problem += aux_W5_Z1 >= W5_UB * Z1 + W5 * Z1_UB - W5_UB * Z1_UB
problem += aux_W5_Z1 <= W5_UB * Z1 + W5 * Z1_LB - W5_UB * Z1_LB
problem += aux_W5_Z1 <= W5_LB * Z1 + W5 * Z1_UB - W5_LB * Z1_UB

# McCormick Constraints for aux_W6_Z2 = W6 * Z2
problem += aux_W6_Z2 >= W6_LB * Z2 + W6 * Z2_LB - W6_LB * Z2_LB
problem += aux_W6_Z2 >= W6_UB * Z2 + W6 * Z2_UB - W6_UB * Z2_UB
problem += aux_W6_Z2 <= W6_UB * Z2 + W6 * Z2_LB - W6_UB * Z2_LB
problem += aux_W6_Z2 <= W6_LB * Z2 + W6 * Z2_UB - W6_LB * Z2_UB

# Output Layer Computation using auxiliary variables
problem += Z3 == aux_W5_Z1 + aux_W6_Z2 + b3

# Classification Decision
problem += y_pred >= 0
problem += y_pred <= 1
problem += y_pred >= (Z3 >= 0)
problem += y_pred <= (Z3 >= 0) + 1

# Loss function (absolute error)
loss = LpVariable("loss", lowBound=0)
problem += loss >= y_pred - y
problem += loss >= y - y_pred

# Objective: Minimize loss
problem += loss

# Solve the LP
problem.solve()

# Print results
print(f"Predicted y = {value(y_pred)}, Actual y = {y}")
print(f"Optimal Weights: W1={value(W1)}, W2={value(W2)}, W3={value(W3)}, W4={value(W4)}, W5={value(W5)}, W6={value(W6)}")
print(f"Optimal Biases: b1={value(b1)}, b2={value(b2)}, b3={value(b3)}")
