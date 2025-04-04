import gurobipy as gp
from gurobipy import GRB
import numpy as np

X = np.array([
    [2, 3], [4, 1], [5, -2], [-3, 6], [7, -4],
    [0, 2], [-1, 5], [3, -3], [2, -5], [4, 0]
])
W1 = np.array([[1, -1], [2, 3]])
W2 = np.array([[0.5], [-1.5]])
# y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
y = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1])

model = gp.Model("Minimize_b")

b1 = model.addVars(2, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b1")
b2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b2")

abs_b1 = model.addVars(2, lb=0, name="abs_b1")
abs_b2 = model.addVar(lb=0, name="abs_b2")

for j in range(2):
    model.addConstr(b1[j] <= abs_b1[j], f"abs_b1_{j}_upper")
    model.addConstr(-b1[j] <= abs_b1[j], f"abs_b1_{j}_lower")
model.addConstr(b2 <= abs_b2, f"abs_b2_upper")
model.addConstr(-b2 <= abs_b2, f"abs_b2_lower")

Z1 = model.addVars(10, 2, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z1") 
A1 = model.addVars(10, 2, lb=0, name="A1") 

# for i in range(10):
#     for j in range(2):
#         Z1[i, j] = X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j]
#         model.addConstr(A1[i, j] >= Z1[i, j], f"ReLU_{i}_{j}_pos")
#         model.addConstr(A1[i, j] >= 0, f"ReLU_{i}_{j}_zero")


for i in range(10):
    for j in range(2):
        # Z1[i, j] = X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j]
        model.addConstr(Z1[i, j] == X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j], f"Z1_def_{i}_{j}")

        z = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        # M = 100
        M=max(np.abs(X @ W1).flatten()) + 1
        print("m", M)
        model.addConstr(A1[i, j] >= Z1[i, j], f"ReLU_{i}_{j}_pos")
        model.addConstr(A1[i, j] >= 0, f"ReLU_{i}_{j}_zero")
        model.addConstr(A1[i, j] <= Z1[i, j] + M * (1 - z), f"ReLU_{i}_{j}_BigM1")
        model.addConstr(A1[i, j] <= M * z, f"ReLU_{i}_{j}_BigM2")

Z2 = model.addVars(10, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z2") 

for i in range(10):
    # Z2[i] = sum(A1[i, j] * W2[j, 0] for j in range(2)) + b2
    model.addConstr(Z2[i] == sum(A1[i, j] * W2[j, 0] for j in range(2)) + b2, f"Z2_def_{i}")

    # model.addConstr(Z2[i] >= -10, f"Z2_{i}_positive")
    # model.addConstr(Z2[i] <= 10, f"Z2_{i}_positive")

    if y[i] == 1:
        model.addConstr(Z2[i] >= 0, f"Z2_{i}_positive")
    else:
        model.addConstr(Z2[i] <= -0.01, f"Z2_{i}_negative")
    # if i == 1:
    #     model.addConstr(Z2[i] <= 0.99999, f"Z2_{i}_positive")


model.setObjective(
    gp.quicksum(Z2[i] for i in range(10) if y[i] == 1) - gp.quicksum(Z2[i] for i in range(10) if y[i] == 0) + 
    0.01 * (b1[0] * b1[0] + b1[1] * b1[1] + b2 * b2),
    GRB.MINIMIZE
)

# model.setObjective(b1[0] * b1[0] + b1[1] * b1[1] + b2 * b2, GRB.MINIMIZE)
# model.setObjectiveN(abs_b1[0], index=0, priority=1, name="Minimize_b1_0")
# model.setObjectiveN(abs_b1[1], index=1, priority=1, name="Minimize_b1_1")

model.optimize()

if model.status == GRB.OPTIMAL:
    b1_values = [b1[i].X for i in range(2)]
    b2_value = b2.X
    print("Optimal solution found:")
    print(f"b1: {b1_values}")
    print(f"b2: {b2_value}")

    Z1_values = [
        [X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j].X for j in range(2)]
        for i in range(10)
    ]
    print("Z1", Z1_values)

    A1_values = [[A1[i, j].X for j in range(2)] for i in range(10)]
    print("A1", A1_values)

    

    Z2_values = [A1_values[i][0] * W2[0, 0] + A1_values[i][1] * W2[1, 0] + b2.X for i in range(10)]
    print("Z2", Z2_values)
    
else:
    print("No feasible solution found.")
