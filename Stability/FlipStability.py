from Gurobi_ForwardPass import ForwardPass_L2_ReLU, ForwardPass_L2_Sigmoid
from Weights_VerifyQuantify_L2 import VerifyWeights as VerifyWeights_L2
import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 60

def RunForward_L2(nn, X, y, activation, tol, n, flipCount, l1, l2):
    l1_size = nn.W[0].shape[1]
    l2_size = nn.W[1].shape[1]
    l3_size = nn.W[2].shape[1]

    model = gp.Model("Minimize_b")

    W1_offset = model.addVars(nn.W[0].shape[0], l1_size, vtype=GRB.CONTINUOUS, name="W1_offset")
    W2_offset = model.addVars(nn.W[1].shape[0], l2_size, vtype=GRB.CONTINUOUS, name="W2_offset")
    W3_offset = model.addVars(nn.W[2].shape[0], l3_size, vtype=GRB.CONTINUOUS, name="W3_offset")

    b1_offset = model.addVars(l1_size, vtype=GRB.CONTINUOUS, name="b1_offset")
    b2_offset = model.addVars(l2_size, vtype=GRB.CONTINUOUS, name="b2_offset")
    b3_offset = model.addVars(l3_size, vtype=GRB.CONTINUOUS, name="b3_offset")

    NewW1 = [[nn.W[0][i][j] + W1_offset[i, j] for j in range(l1_size)] for i in range(nn.W[0].shape[0])]
    NewW2 = [[nn.W[1][i][j] + W2_offset[i, j] for j in range(l2_size)] for i in range(nn.W[1].shape[0])]
    NewW3 = [[nn.W[2][i][j] + W3_offset[i, j] for j in range(l3_size)] for i in range(nn.W[2].shape[0])]
    Newb1 = [[nn.b[0][0, i] + b1_offset[i] for i in range(l1_size)]]
    Newb2 = [[nn.b[1][0, i] + b2_offset[i] for i in range(l2_size)]]
    Newb3 = [[nn.b[2][0, i] + b3_offset[i] for i in range(l3_size)]]

    if activation == "relu":
        Z3 = ForwardPass_L2_ReLU(model, X, NewW1, NewW2, NewW3, Newb1, Newb2, Newb3)
    elif activation == "sigmoid":
        Z3 = ForwardPass_L2_Sigmoid(model, X, NewW1, NewW2, NewW3, Newb1, Newb2, Newb3)

    y_g = [model.addVar(vtype=GRB.BINARY, name=f"y2_{i}") for i in range(len(X))]
    f = model.addVars(len(X), vtype=GRB.BINARY, name=f"flip_i") 

    M = 50
    model.addConstr(sum(f[i] for i in range(len(X))) == flipCount, "one_flip")

    E = tol
    for i in range(len(X)):
        y_scalar = int(y[i])
        model.addConstr(Z3[i, 0] >= E - M * (1 - y_g[i]), f"Z3_{i}_lower_bound")
        model.addConstr(Z3[i, 0] <= -E + M * y_g[i], f"Z3_{i}_upper_bound")
        model.addConstr(f[i] >= y_g[i] - y_scalar, f"flip_upper_{i}")
        model.addConstr(f[i] >= y_scalar - y_g[i], f"flip_lower_{i}")
        model.addConstr(f[i] <= y_g[i] + y_scalar, f"flip_cap_{i}")
        model.addConstr(f[i] <= 2 - y_g[i] - y_scalar, f"flip_cap_2_{i}")

    abs_b1 = model.addVars(l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b1")
    abs_b2 = model.addVars(l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b2")
    abs_b3 = model.addVars(l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b3")
    abs_W1 = model.addVars(nn.W[0].shape[0], l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W1")
    abs_W2 = model.addVars(nn.W[1].shape[0], l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W2")
    abs_W3 = model.addVars(nn.W[2].shape[0], l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W3")

    for i in range(l1_size):
        model.addConstr(abs_b1[i] >= b1_offset[i])
        model.addConstr(abs_b1[i] >= -b1_offset[i])
    for i in range(l2_size):
        model.addConstr(abs_b2[i] >= b2_offset[i])
        model.addConstr(abs_b2[i] >= -b2_offset[i])
    for i in range(l3_size):
        model.addConstr(abs_b3[i] >= b3_offset[i])
        model.addConstr(abs_b3[i] >= -b3_offset[i])
    for i in range(nn.W[0].shape[0]):
        for j in range(l1_size):
            model.addConstr(abs_W1[i, j] >= W1_offset[i, j])
            model.addConstr(abs_W1[i, j] >= -W1_offset[i, j])
    for i in range(nn.W[1].shape[0]):
        for j in range(l2_size):
            model.addConstr(abs_W2[i, j] >= W2_offset[i, j])
            model.addConstr(abs_W2[i, j] >= -W2_offset[i, j])
    for i in range(nn.W[2].shape[0]):
        for j in range(l3_size):
            model.addConstr(abs_W3[i, j] >= W3_offset[i, j])
            model.addConstr(abs_W3[i, j] >= -W3_offset[i, j])

    objective = (
        gp.quicksum(abs_b1[i] for i in range(l1_size)) +
        gp.quicksum(abs_b2[i] for i in range(l2_size)) +
        gp.quicksum(abs_b3[i] for i in range(l3_size)) +
        gp.quicksum(abs_W1[i, j] for i in range(nn.W[0].shape[0]) for j in range(l1_size)) +
        gp.quicksum(abs_W2[i, j] for i in range(nn.W[1].shape[0]) for j in range(l2_size)) +
        gp.quicksum(abs_W3[i, j] for i in range(nn.W[2].shape[0]) for j in range(l3_size))
    )

    model.setObjective(objective, GRB.MINIMIZE)
    model.setParam("FeasibilityTol", 1e-9)
    model.addConstr(objective >= 0, "NonNegativeObjective")
    model.setParam('TimeLimit', timeLimit)
    model.optimize()
    
    if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
        if model.SolCount == 0:
            return 0, 0, 0, -1

        f_values = [f[i].X for i in range(len(X))]
        y_g_values = [y_g[i].X for i in range(len(X))]
        flip_idxs = [i for i, val in enumerate(f_values) if val == 1]

        if len(flip_idxs) != flipCount:
            return 0, 0, 0, len(flip_idxs)

        Z3_values = [[Z3[i, j].X for j in range(l3_size)] for i in range(len(X))]

        W1_values = np.array([[nn.W[0][i][j] for j in range(l1_size)] for i in range(nn.W[0].shape[0])])
        W2_values = np.array([[nn.W[1][i][j] for j in range(l2_size)] for i in range(nn.W[1].shape[0])])
        W3_values = np.array([[nn.W[2][i][j] for j in range(l3_size)] for i in range(nn.W[2].shape[0])])
        b1_values = np.array([nn.b[0][0, j] for j in range(l1_size)])
        b2_values = np.array([nn.b[1][0, j] for j in range(l2_size)])
        b3_values = np.array([nn.b[2][0, j] for j in range(l3_size)])

        W1_values_with_offset = np.array([[nn.W[0][i][j] + W1_offset[i, j].X for j in range(l1_size)] for i in range(nn.W[0].shape[0])])
        W2_values_with_offset = np.array([[nn.W[1][i][j] + W2_offset[i, j].X for j in range(l2_size)] for i in range(nn.W[1].shape[0])])
        W3_values_with_offset = np.array([[nn.W[2][i][j] + W3_offset[i, j].X for j in range(l3_size)] for i in range(nn.W[2].shape[0])])
        b1_values_with_offset = np.array([nn.b[0][0, j] + b1_offset[j].X for j in range(l1_size)])
        b2_values_with_offset = np.array([nn.b[1][0, j] + b2_offset[j].X for j in range(l2_size)])
        b3_values_with_offset = np.array([nn.b[2][0, j] + b3_offset[j].X for j in range(l3_size)])

        vw = VerifyWeights_L2(n, l1, l2, flip_idxs, tol, W1_values, W2_values, W3_values, b1_values, b2_values, b3_values,
                              W1_values_with_offset, W2_values_with_offset, W3_values_with_offset,
                              b1_values_with_offset, b2_values_with_offset, b3_values_with_offset)
        mismatch, max_abs_value, mean_value, sum_abs_value = vw.main(X, activation)

        return max_abs_value, mean_value, sum_abs_value, mismatch
    else:
        return 0, 0, 0, -1
