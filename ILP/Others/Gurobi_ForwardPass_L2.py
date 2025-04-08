import gurobipy as gp
from gurobipy import GRB

def ForwardPass(model, X, W1, W2, W3, b1, b2, b3):
    l1_size = len(W1[0])
    l2_size = len(W2[0])
    l3_size = len(W3[0])


    Z1 = model.addVars(len(X), l1_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z1") 
    A1 = model.addVars(len(X), l1_size, lb=0, name="A1") 

    M = 100
    # M = model.addVar(vtype=GRB.CONTINUOUS, name="M")  # Big-M variable

    # for j in range(l1_size):
    #     for row_idx in range(len(X)):  # Loop through all samples
    #         model.addConstr(M >= gp.quicksum(abs(X[row_idx][i]) * W1[i][j] for i in range(len(X[0]))) + b1[0][j],
    #                         f"BigM_LB_{row_idx}_{j}")
    z1 = model.addVars(len(X), l1_size, vtype=GRB.BINARY, name="z1")
    z2 = model.addVars(len(X), l2_size, vtype=GRB.BINARY, name="z2")
    z3 = model.addVars(len(X), l3_size, vtype=GRB.BINARY, name="z3")

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        for j in range(l1_size):
            model.addConstr(
                Z1[row_idx, j] == sum(X_row[i] * W1[i][j] for i in range(len(X_row))) + b1[0][j],
                f"Z1_def_{row_idx}_{j}"
            )

            model.addConstr(A1[row_idx, j] >= Z1[row_idx, j], f"ReLU_{row_idx}_{j}_pos")
            model.addConstr(A1[row_idx, j] >= 0, f"ReLU_{row_idx}_{j}_zero")
            model.addConstr(A1[row_idx, j] <= Z1[row_idx, j] + M * (1 - z1[row_idx, j]), f"ReLU_{row_idx}_{j}_BigM1")
            model.addConstr(A1[row_idx, j] <= M * z1[row_idx, j], f"ReLU_{row_idx}_{j}_BigM2")

    Z2 = model.addVars(len(X), l2_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z2") 
    A2 = model.addVars(len(X), l2_size, lb=0, name="A2") 

    for row_idx in range(len(X)):
        for j in range(l2_size):
            model.addConstr(
                Z2[row_idx, j] == sum(A1[row_idx, i] * W2[i][j] for i in range(l1_size)) + b2[0][j],
                f"Z2_def_{row_idx}_{j}"
            )
            model.addConstr(A2[row_idx, j] >= Z2[row_idx, j], f"ReLU_A2_{row_idx}_{j}_pos")
            model.addConstr(A2[row_idx, j] >= 0, f"ReLU_A2_{row_idx}_{j}_zero")
            model.addConstr(A2[row_idx, j] <= Z2[row_idx, j] + M * (1 - z2[row_idx, j]), f"ReLU_A2_{row_idx}_{j}_BigM1")
            model.addConstr(A2[row_idx, j] <= M * z2[row_idx, j], f"ReLU_A2_{row_idx}_{j}_BigM2")

    Z3 = model.addVars(len(X), l3_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z3")

    for row_idx in range(len(X)):
        for j in range(l3_size):
            model.addConstr(
                Z3[row_idx, j] == sum(A2[row_idx, i] * W3[i][j] for i in range(l2_size)) + b3[0][j],
                f"Z3_def_{row_idx}_{j}"
            )

    return Z3
