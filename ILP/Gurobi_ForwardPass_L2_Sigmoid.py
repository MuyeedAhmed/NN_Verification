import gurobipy as gp
from gurobipy import GRB

def ForwardPass(model, X, W1, W2, W3, b1, b2, b3):
    model.setParam('NonConvex', 2)

    l1_size = len(W1[0])
    l2_size = len(W2[0])
    l3_size = len(W3[0])

    Z1 = model.addVars(len(X), l1_size, lb=-GRB.INFINITY, name="Z1") 
    A1 = model.addVars(len(X), l1_size, lb=0, ub=1, name="A1")

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        for j in range(l1_size):
            model.addConstr(
                Z1[row_idx, j] == sum(X_row[i] * W1[i][j] for i in range(len(X_row))) + b1[0][j],
                f"Z1_def_{row_idx}_{j}"
            )
            model.addGenConstrLogistic(Z1[row_idx, j], A1[row_idx, j], name=f"Sigmoid1_{row_idx}_{j}")

    Z2 = model.addVars(len(X), l2_size, lb=-GRB.INFINITY, name="Z2") 
    A2 = model.addVars(len(X), l2_size, lb=0, ub=1, name="A2")

    for row_idx in range(len(X)):
        for j in range(l2_size):
            model.addConstr(
                Z2[row_idx, j] == sum(A1[row_idx, i] * W2[i][j] for i in range(l1_size)) + b2[0][j],
                f"Z2_def_{row_idx}_{j}"
            )
            model.addGenConstrLogistic(Z2[row_idx, j], A2[row_idx, j], name=f"Sigmoid2_{row_idx}_{j}")

    Z3 = model.addVars(len(X), l3_size, lb=-GRB.INFINITY, name="Z3")

    for row_idx in range(len(X)):
        for j in range(l3_size):
            model.addConstr(
                Z3[row_idx, j] == sum(A2[row_idx, i] * W3[i][j] for i in range(l2_size)) + b3[0][j],
                f"Z3_def_{row_idx}_{j}"
            )

    return Z3
