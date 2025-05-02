import gurobipy as gp
from gurobipy import GRB

def ForwardPass_L2_ReLU(model, X, W1, W2, W3, b1, b2, b3):
    l1_size = len(W1[0])
    l2_size = len(W2[0])
    l3_size = len(W3[0])


    Z1 = model.addVars(len(X), l1_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z1") 
    A1 = model.addVars(len(X), l1_size, lb=0, name="A1") 
    z1 = model.addVars(len(X), l1_size, vtype=GRB.BINARY, name="z1")
    z2 = model.addVars(len(X), l2_size, vtype=GRB.BINARY, name="z2")
    

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        for j in range(l1_size):
            model.addConstr(
                Z1[row_idx, j] == sum(X_row[i] * W1[i][j] for i in range(len(X_row))) + b1[0][j],
                f"Z1_def_{row_idx}_{j}"
            )

            model.addConstr((z1[row_idx, j] == 1) >> (Z1[row_idx, j] >= 0), name=f"ReLU_pos_{row_idx}_{j}")
            model.addConstr((z1[row_idx, j] == 1) >> (A1[row_idx, j] == Z1[row_idx, j]), name=f"ReLU_eq_{row_idx}_{j}")

            model.addConstr((z1[row_idx, j] == 0) >> (Z1[row_idx, j] <= 0), name=f"ReLU_neg_{row_idx}_{j}")
            model.addConstr((z1[row_idx, j] == 0) >> (A1[row_idx, j] == 0), name=f"ReLU_zero_{row_idx}_{j}")


    Z2 = model.addVars(len(X), l2_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z2") 
    A2 = model.addVars(len(X), l2_size, lb=0, name="A2") 

    for row_idx in range(len(X)):
        for j in range(l2_size):
            model.addConstr(
                Z2[row_idx, j] == sum(A1[row_idx, i] * W2[i][j] for i in range(l1_size)) + b2[0][j],
                f"Z2_def_{row_idx}_{j}"
            )

            model.addConstr((z2[row_idx, j] == 1) >> (Z2[row_idx, j] >= 0), name=f"ReLU2_pos_{row_idx}_{j}")
            model.addConstr((z2[row_idx, j] == 1) >> (A2[row_idx, j] == Z2[row_idx, j]), name=f"ReLU2_eq_{row_idx}_{j}")

            model.addConstr((z2[row_idx, j] == 0) >> (Z2[row_idx, j] <= 0), name=f"ReLU2_neg_{row_idx}_{j}")
            model.addConstr((z2[row_idx, j] == 0) >> (A2[row_idx, j] == 0), name=f"ReLU2_zero_{row_idx}_{j}")

    Z3 = model.addVars(len(X), l3_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z3")

    for row_idx in range(len(X)):
        for j in range(l3_size):
            model.addConstr(
                Z3[row_idx, j] == sum(A2[row_idx, i] * W3[i][j] for i in range(l2_size)) + b3[0][j],
                f"Z3_def_{row_idx}_{j}"
            )

    return Z3

def ForwardPass_L2_Sigmoid(model, X, W1, W2, W3, b1, b2, b3):
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

def ForwardPass_L3_ReLU(model, X, W1, W2, W3, W4, b1, b2, b3, b4):
    l1_size = len(W1[0])
    l2_size = len(W2[0])
    l3_size = len(W3[0])
    l4_size = len(W4[0])

    Z1 = model.addVars(len(X), l1_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z1")
    A1 = model.addVars(len(X), l1_size, lb=0, name="A1")
    z1 = model.addVars(len(X), l1_size, vtype=GRB.BINARY, name="z1")

    Z2 = model.addVars(len(X), l2_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z2")
    A2 = model.addVars(len(X), l2_size, lb=0, name="A2")
    z2 = model.addVars(len(X), l2_size, vtype=GRB.BINARY, name="z2")

    Z3 = model.addVars(len(X), l3_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z3")
    A3 = model.addVars(len(X), l3_size, lb=0, name="A3")
    z3 = model.addVars(len(X), l3_size, vtype=GRB.BINARY, name="z3")

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        for j in range(l1_size):
            model.addConstr(Z1[row_idx, j] == sum(X_row[i] * W1[i][j] for i in range(len(X_row))) + b1[0][j])
            model.addConstr((z1[row_idx, j] == 1) >> (Z1[row_idx, j] >= 0))
            model.addConstr((z1[row_idx, j] == 1) >> (A1[row_idx, j] == Z1[row_idx, j]))
            model.addConstr((z1[row_idx, j] == 0) >> (Z1[row_idx, j] <= 0))
            model.addConstr((z1[row_idx, j] == 0) >> (A1[row_idx, j] == 0))

        for j in range(l2_size):
            model.addConstr(Z2[row_idx, j] == sum(A1[row_idx, i] * W2[i][j] for i in range(l1_size)) + b2[0][j])
            model.addConstr((z2[row_idx, j] == 1) >> (Z2[row_idx, j] >= 0))
            model.addConstr((z2[row_idx, j] == 1) >> (A2[row_idx, j] == Z2[row_idx, j]))
            model.addConstr((z2[row_idx, j] == 0) >> (Z2[row_idx, j] <= 0))
            model.addConstr((z2[row_idx, j] == 0) >> (A2[row_idx, j] == 0))

        for j in range(l3_size):
            model.addConstr(Z3[row_idx, j] == sum(A2[row_idx, i] * W3[i][j] for i in range(l2_size)) + b3[0][j])
            model.addConstr((z3[row_idx, j] == 1) >> (Z3[row_idx, j] >= 0))
            model.addConstr((z3[row_idx, j] == 1) >> (A3[row_idx, j] == Z3[row_idx, j]))
            model.addConstr((z3[row_idx, j] == 0) >> (Z3[row_idx, j] <= 0))
            model.addConstr((z3[row_idx, j] == 0) >> (A3[row_idx, j] == 0))

    Z4 = model.addVars(len(X), l4_size, lb=-GRB.INFINITY, name="Z4")
    for row_idx in range(len(X)):
        for j in range(l4_size):
            model.addConstr(Z4[row_idx, j] == sum(A3[row_idx, i] * W4[i][j] for i in range(l3_size)) + b4[0][j])
    return Z4


def ForwardPass_L3_Sigmoid(model, X, W1, W2, W3, W4, b1, b2, b3, b4):
    model.setParam('NonConvex', 2)

    l1_size = len(W1[0])
    l2_size = len(W2[0])
    l3_size = len(W3[0])
    l4_size = len(W4[0])

    Z1 = model.addVars(len(X), l1_size, lb=-GRB.INFINITY, name="Z1")
    A1 = model.addVars(len(X), l1_size, lb=0, ub=1, name="A1")

    Z2 = model.addVars(len(X), l2_size, lb=-GRB.INFINITY, name="Z2")
    A2 = model.addVars(len(X), l2_size, lb=0, ub=1, name="A2")

    Z3 = model.addVars(len(X), l3_size, lb=-GRB.INFINITY, name="Z3")
    A3 = model.addVars(len(X), l3_size, lb=0, ub=1, name="A3")

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        for j in range(l1_size):
            model.addConstr(Z1[row_idx, j] == sum(X_row[i] * W1[i][j] for i in range(len(X_row))) + b1[0][j])
            model.addGenConstrLogistic(Z1[row_idx, j], A1[row_idx, j])

        for j in range(l2_size):
            model.addConstr(Z2[row_idx, j] == sum(A1[row_idx, i] * W2[i][j] for i in range(l1_size)) + b2[0][j])
            model.addGenConstrLogistic(Z2[row_idx, j], A2[row_idx, j])

        for j in range(l3_size):
            model.addConstr(Z3[row_idx, j] == sum(A2[row_idx, i] * W3[i][j] for i in range(l2_size)) + b3[0][j])
            model.addGenConstrLogistic(Z3[row_idx, j], A3[row_idx, j])

    Z4 = model.addVars(len(X), l4_size, lb=-GRB.INFINITY, name="Z4")
    for row_idx in range(len(X)):
        for j in range(l4_size):
            model.addConstr(Z4[row_idx, j] == sum(A3[row_idx, i] * W4[i][j] for i in range(l3_size)) + b4[0][j])
    return Z4
