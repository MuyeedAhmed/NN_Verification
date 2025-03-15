import gurobipy as gp
from gurobipy import GRB
import numpy as np

X = np.array([
    [2, 3], [4, 1], [5, -2], [-3, 6], [7, -4],
    [0, 2], [-1, 5], [3, -3], [2, -5], [4, 0]
])
W1 = np.array([[1, -1], [2, 3]])
y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

model = gp.Model("reduce_b1")

b1 = model.addVars(2, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b1")

abs_b1 = model.addVars(2, lb=0, name="abs_b1")

for j in range(2):
    model.addConstr(b1[j] <= abs_b1[j], f"abs_b1_{j}_upper")
    model.addConstr(-b1[j] <= abs_b1[j], f"abs_b1_{j}_lower")

for i in range(10):
    for j in range(2):
        Z1_ij = X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j]
        if i == 4 and j == 1:
            model.addConstr(Z1_ij <= -1e-16, f"Z1_{i}_{j}_constraint")
        else:
            model.addConstr(Z1_ij >= 0, f"Z1_{i}_{j}_constraint")
        

model.setObjectiveN(abs_b1[0], index=0, priority=1, name="Minimize_b1_0")
model.setObjectiveN(abs_b1[1], index=1, priority=1, name="Minimize_b1_1")

model.optimize()

if model.status == GRB.OPTIMAL:
    b1_values = [b1[j].x for j in range(2)]
    print("Optimal b1 values:", b1_values)
