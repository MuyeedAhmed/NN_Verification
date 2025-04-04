import gurobipy as gp
from gurobipy import GRB
import numpy as np

X = np.array([
    [2, 3], [4, 1], [5, -2], [-3, 6], [7, -4],
    [0, 2], [-1, 5], [3, -3], [2, -5], [4, 0]
])
W1 = np.array([[1, -1], [2, 3]])
W2 = np.array([[0.5], [-1.5]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

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

Z1 = model.addVars(10, 2, lb=0, name="Z1") 
A1 = model.addVars(10, 2, lb=0, name="A1") 

for i in range(10):
    for j in range(2):
        Z1[i, j] = X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j]
        model.addConstr(A1[i, j] >= Z1[i, j], f"ReLU_{i}_{j}_pos")
        model.addConstr(A1[i, j] >= 0, f"ReLU_{i}_{j}_zero")
        
for i in range(10):
    Z2_i = sum(A1[i, j] * W2[j, 0] for j in range(2)) + b2
    if y[i] == 1:
        model.addConstr(Z2_i >= 0, f"Z2_{i}_positive")
    else:
        model.addConstr(Z2_i <= 0.0000000001, f"Z2_{i}_negative")


# model.setObjective(b1[0] * b1[0] + b1[1] * b1[1] + b2 * b2, GRB.MINIMIZE)
model.setObjectiveN(abs_b1[0], index=0, priority=1, name="Minimize_b1_0")
model.setObjectiveN(abs_b1[1], index=1, priority=1, name="Minimize_b1_1")

model.optimize()

if model.status == GRB.OPTIMAL:
    b1_values = [b1[i].X for i in range(2)]
    b2_value = b2.X
    print("Optimal solution found:")
    print(f"b1: {b1_values}")
    print(f"b2: {b2_value}")

    A1_values = [[A1[i, j].X for j in range(2)] for i in range(10)]
    print(A1_values)

    Z1_values = [
        [X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j].X for j in range(2)]
        for i in range(10)
    ]
    print(Z1_values)
else:
    print("No feasible solution found.")
