import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch

timeLimit = 3000

def FlipBinary(file_name, X, y, y_gt, W, b, n_samples, tol, flipCount):
    if n_samples == -1:
        n_samples = X.shape[0]
    y = y.reshape(-1, 1)[:n_samples]
    X = X[:n_samples]
    

    W_size = W.shape[1]
    Output_size = 1

    model = gp.Model("Flip")
    W_offset = model.addVars(Output_size, W_size, vtype=GRB.CONTINUOUS, name="W_offset")
    b_offset = model.addVars(W_size, vtype=GRB.CONTINUOUS, name="b_offset")
    
    NewW = [[W[i][j] + W_offset[i, j] for j in range(W_size)] for i in range(Output_size)]
    Newb = [b[i] + b_offset[i] for i in range(Output_size)]

    Z = model.addVars(n_samples, W_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z")

    for r in range(n_samples):
        for j in range(Output_size):
            model.addConstr(
                Z[r, j] == gp.quicksum(X[r][i] * NewW[j][i] for i in range(W_size)) + Newb[j],
                name=f"Z_def_{r}_{j}"
            )
    
    mismatch_flag = model.addVars(n_samples, vtype=GRB.BINARY, name="mismatch_flag")
    for i in range(n_samples):
        y_scalar = int(y[i])
        if y_scalar == 1:
            model.addConstr((mismatch_flag[i] == 1) >> (Z[i, 0] <= -tol), f"Z_{i}_mismatch")
            model.addConstr((mismatch_flag[i] == 0) >> (Z[i, 0] >= tol), f"Z_{i}_match")
        else:
            model.addConstr((mismatch_flag[i] == 1) >> (Z[i, 0] >= tol), f"Z_{i}_mismatch")
            model.addConstr((mismatch_flag[i] == 0) >> (Z[i, 0] <= -tol), f"Z_{i}_match")

    model.addConstr(gp.quicksum(mismatch_flag[i] for i in range(n_samples)) == flipCount, "m_flips")    

    abs_b = model.addVars(W_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b")
    abs_W = model.addVars(Output_size, W_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W")
    
    for i in range(W_size):
        model.addConstr(abs_b[i] >= b_offset[i])
        model.addConstr(abs_b[i] >= -b_offset[i])

    for i in range(Output_size):
        for j in range(W_size):
            model.addConstr(abs_W[i, j] >= W_offset[i, j])
            model.addConstr(abs_W[i, j] >= -W_offset[i, j])

    objective = (
        gp.quicksum(abs_b[i] for i in range(W_size)) +
        gp.quicksum(abs_W[i, j] for i in range(Output_size) for j in range(W_size))
    )


    model.setObjective(objective, GRB.MINIMIZE)
    model.addConstr(objective >= 0, "NonNegativeObjective")
    model.setParam('TimeLimit', timeLimit)
    model.optimize()

    if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
        if model.SolCount == 0:
            print("Timeout")
            return False
        f_values = [mismatch_flag[i].X for i in range(n_samples)]
        flip_idxs = [i for i, val in enumerate(f_values) if val == 1]
        Z_values = np.array([[Z[i, j].X for j in range(W_size)] for i in range(n_samples)])

        if len(flip_idxs) != flipCount:
            print(f"Error: Expected {flipCount} value of 1 in f_values, but found {len(flip_idxs)}")
            return None
        
        W_values_with_offset = np.array([[W[i][j] + W_offset[i, j].X for j in range(W_size)] for i in range(Output_size)])
        b_values_with_offset = np.array([b[j] + b_offset[j].X for j in range(Output_size)])

        return [W_values_with_offset, b_values_with_offset]
    else:
        print("No feasible solution found.")
        return None

def FlipMulticlass(file_name, X, y, y_gt, W, b, n_samples, tol, flipCount):
    pass
def BorderBinary(file_name, X, y, y_gt, W, b, n_samples, tol):
    # n_samples = 10
    if n_samples == -1:
        n_samples = X.shape[0]
    y = y.reshape(-1, 1)[:n_samples]
    X = X[:n_samples]
    

    W_size = W.shape[1]
    Output_size = 1

    model = gp.Model("Flip")
    W_offset = model.addVars(Output_size, W_size, vtype=GRB.CONTINUOUS, name="W_offset")
    b_offset = model.addVars(W_size, vtype=GRB.CONTINUOUS, name="b_offset")
    
    NewW = [[W[i][j] + W_offset[i, j] for j in range(W_size)] for i in range(Output_size)]
    Newb = [b[i] + b_offset[i] for i in range(Output_size)]

    Z = model.addVars(n_samples, Output_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z")

    for r in range(n_samples):
        for j in range(Output_size):
            model.addConstr(
                Z[r, j] == gp.quicksum(X[r][i] * NewW[j][i] for i in range(W_size)) + Newb[j],
                name=f"Z_def_{r}_{j}"
            )
    abs_Z_vars = []
    for i in range(n_samples):
        y_scalar = int(y[i])
        if y_scalar == 1:
            model.addConstr(Z[i, 0] >= tol, f"Z_{i}_positive")
        else:
            model.addConstr(Z[i, 0] <= -tol, f"Z_{i}_negative")

        abs_Z = model.addVar(lb=0.0, name=f"abs_Z_{i}")
        model.addConstr(abs_Z >= Z[i, 0], name=f"abs_upper_{i}")
        model.addConstr(abs_Z >= -Z[i, 0], name=f"abs_lower_{i}")
        abs_Z_vars.append(abs_Z)

    model.setObjective(gp.quicksum(abs_Z_vars), GRB.MINIMIZE)
    # for i in range(n_samples):
    #     y_scalar = int(y[i])
    #     if y_scalar == 1:
    #         model.addConstr(Z[i, 0] >= tol, f"Z_{i}_positive")
    #     else:
    #         model.addConstr(Z[i, 0] <= -tol, f"Z_{i}_negative")

    # abs_diffs = []
    # for i in range(len(X)):
    #     diff = model.addVar(lb=0.0, name=f"abs_diff_{i}")
    #     model.addConstr(diff >= Z[i, 0] - int(y[i]), name=f"abs_upper_{i}")
    #     model.addConstr(diff >= int(y[i]) - Z[i, 0], name=f"abs_lower_{i}")
    #     abs_diffs.append(diff)
    # l1_loss = gp.quicksum(abs_diffs)
    # model.addConstr(l1_loss <= 10000, "ObjectiveUpperBound")
    # # model.addConstr(l1_loss >= 1, "ObjectiveNonNegative")
    # model.setObjective(l1_loss, GRB.MAXIMIZE)

    # model.setParam('TimeLimit', timeLimit)
    model.optimize()

    if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
        if model.SolCount == 0:
            print("Timeout")
            return False
        # Z_values = np.array([[Z[i, j].X for j in range(Output_size)] for i in range(n_samples)])
        # print("Z_values:", Z_values)
        # print(y)
        
        W_values_with_offset = np.array([[W[i][j] + W_offset[i, j].X for j in range(W_size)] for i in range(Output_size)])
        b_values_with_offset = np.array([b[j] + b_offset[j].X for j in range(Output_size)])

        return [W_values_with_offset, b_values_with_offset]
    else:
        print("No feasible solution found.")
        return None


def BorderMulticlass(file_name, X, y, y_gt, W, b, n_samples, tol):
    pass

