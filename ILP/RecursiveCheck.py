from NN.Network_PresetWeights import NN_preset, RunNN_preset
from Gurobi_ForwardPass_L2_Indicator import ForwardPass
from Weights.QuantifyVerifyWeights_L2 import VerifyWeights

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import subprocess

timeLimit = 6000
accuracy_file = "accuracy.txt"
def main():
    n = 106
    l1 = 4
    l2 = 4
    flipCount = 1
    tol = 2e-9

    df = pd.read_csv("../Dataset/appendicitis.csv")
    X = df.iloc[:, :-1].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_true = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    trn = RunNN_preset(X, y_true, hs1=l1, hs2=l2, out_size=1, lr = 0.1, epoch=10000)
    nn, y_predict = trn.TrainReturnWeights()
    # ns = [50, 75, 100, 40, 60]
    # for n in ns:
    X = X[0:n]
    y = y_predict[0:n]
    with open(accuracy_file, "a") as f:
        f.write(f"Iteration: 0,  Accuracy: {np.mean(y_true == y_predict):.4f}\n")

    RunForward(nn, X, y, y_true, tol, n, 1, l1, l2, 0)
    
    # # tolerances = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
    # tolerances = [5e-9, 1e-8, 5e-8]
    # times = []
    # for tol in tolerances:
    #     # for flipCount in range(1, 11):
    #     t0 = time.time()
    #     RunForward(nn, X, y, tol, n, 1, l1, l2)
    #     times.append(time.time()-t0)

    # print("Times:", times)

def RunForward(nn, X, y, y_true, tol, n, flipCount, l1, l2, iter):

    l1_size = len(nn.W1[0])
    l2_size = len(nn.W2[0])
    l3_size = len(nn.W3[0])

    model = gp.Model("Minimize_b")

    W1_offset = model.addVars(len(nn.W1), l1_size, vtype=GRB.CONTINUOUS, name="W1_offset")
    W2_offset = model.addVars(len(nn.W2), l2_size, vtype=GRB.CONTINUOUS, name="W2_offset")
    W3_offset = model.addVars(len(nn.W3), l3_size, vtype=GRB.CONTINUOUS, name="W3_offset")

    b1_offset = model.addVars(l1_size, vtype=GRB.CONTINUOUS, name="b1_offset")
    b2_offset = model.addVars(l2_size, vtype=GRB.CONTINUOUS, name="b2_offset")
    b3_offset = model.addVars(l3_size, vtype=GRB.CONTINUOUS, name="b3_offset")

    NewW1 = [[nn.W1[i][j] + W1_offset[i, j] for j in range(l1_size)] for i in range(len(nn.W1))]
    NewW2 = [[nn.W2[i][j] + W2_offset[i, j] for j in range(l2_size)] for i in range(len(nn.W2))]
    NewW3 = [[nn.W3[i][j] + W3_offset[i, j] for j in range(l3_size)] for i in range(len(nn.W3))]
    Newb1 = [[nn.b1[0, i] + b1_offset[i] for i in range(l1_size)]]
    Newb2 = [[nn.b2[0, i] + b2_offset[i] for i in range(l2_size)]]
    Newb3 = [[nn.b3[0, i] + b3_offset[i] for i in range(l3_size)]]

    Z3 = ForwardPass(model, X, NewW1, NewW2, NewW3, Newb1, Newb2, Newb3)
    
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

        # model.addConstr((y_g[i] == 1) >> (Z3[i, 0] >= 0), name=f"Z3_pos_{i}")
        # model.addConstr((y_g[i] == 0) >> (Z3[i, 0] <= -1e-16), name=f"Z3_neg_{i}")

        
                    
    abs_b1 = model.addVars(l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b1")
    abs_b2 = model.addVars(l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b2")
    abs_b3 = model.addVars(l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b3")

    abs_W1 = model.addVars(len(nn.W1), l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W1")
    abs_W2 = model.addVars(len(nn.W2), l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W2")
    abs_W3 = model.addVars(len(nn.W3), l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W3")

    for i in range(l1_size):
        model.addConstr(abs_b1[i] >= b1_offset[i])
        model.addConstr(abs_b1[i] >= -b1_offset[i])

    for i in range(l2_size):
        model.addConstr(abs_b2[i] >= b2_offset[i])
        model.addConstr(abs_b2[i] >= -b2_offset[i])

    for i in range(l3_size):
        model.addConstr(abs_b3[i] >= b3_offset[i])
        model.addConstr(abs_b3[i] >= -b3_offset[i])

    for i in range(len(nn.W1)):
        for j in range(l1_size):
            model.addConstr(abs_W1[i, j] >= W1_offset[i, j])
            model.addConstr(abs_W1[i, j] >= -W1_offset[i, j])

    for i in range(len(nn.W2)):
        for j in range(l2_size):
            model.addConstr(abs_W2[i, j] >= W2_offset[i, j])
            model.addConstr(abs_W2[i, j] >= -W2_offset[i, j])

    for i in range(len(nn.W3)):
        for j in range(l3_size):
            model.addConstr(abs_W3[i, j] >= W3_offset[i, j])
            model.addConstr(abs_W3[i, j] >= -W3_offset[i, j])

    objective = (
        gp.quicksum(abs_b1[i] for i in range(l1_size)) +
        gp.quicksum(abs_b2[i] for i in range(l2_size)) +
        gp.quicksum(abs_b3[i] for i in range(l3_size)) +
        gp.quicksum(abs_W1[i, j] for i in range(len(nn.W1)) for j in range(l1_size)) +
        gp.quicksum(abs_W2[i, j] for i in range(len(nn.W2)) for j in range(l2_size)) +
        gp.quicksum(abs_W3[i, j] for i in range(len(nn.W3)) for j in range(l3_size))
    )


    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam("FeasibilityTol", 1e-9)
    model.addConstr(objective >= 0, "NonNegativeObjective")
    model.setParam('TimeLimit', timeLimit)
    model.optimize()
    # model.setParam(GRB.Param.NumericFocus, 3)

    # if model.status == GRB.OPTIMAL:
    if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
        if model.SolCount == 0:
            print("Timeout")
            return
        
        f_values = [f[i].X for i in range(len(X))]
        y_g_values = [y_g[i].X for i in range(len(X))]
        flip_idxs = [i for i, val in enumerate(f_values) if val == 1]

        if len(flip_idxs) != flipCount:
            print(f"Error: Expected {flipCount} value of 1 in f_values, but found {len(flip_idxs)}")
            return

        print("f values:", f_values)
        print("y_g values:", y_g_values)

        Z3_values = [[Z3[i, j].X for j in range(l3_size)] for i in range(len(X))]
        print("y_pred:", y.reshape(1,-1)[0])
        print("Z3:", Z3_values)
        
        W1_values = np.array([[nn.W1[i][j] for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_values = np.array([[nn.W2[i][j] for j in range(l2_size)] for i in range(len(nn.W2))])
        W3_values = np.array([[nn.W3[i][j] for j in range(l3_size)] for i in range(len(nn.W3))])
        b1_values = np.array([nn.b1[0, j] for j in range(l1_size)])
        b2_values = np.array([nn.b2[0, j] for j in range(l2_size)])
        b3_values = np.array([nn.b3[0, j] for j in range(l3_size)])
        
        W1_values_with_offset = np.array([[nn.W1[i][j] + W1_offset[i, j].X for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_values_with_offset = np.array([[nn.W2[i][j] + W2_offset[i, j].X for j in range(l2_size)] for i in range(len(nn.W2))])
        W3_values_with_offset = np.array([[nn.W3[i][j] + W3_offset[i, j].X for j in range(l3_size)] for i in range(len(nn.W3))])
        b1_values_with_offset = np.array([nn.b1[0, j] + b1_offset[j].X for j in range(l1_size)])
        b2_values_with_offset = np.array([nn.b2[0, j] + b2_offset[j].X for j in range(l2_size)])
        b3_values_with_offset = np.array([nn.b3[0, j] + b3_offset[j].X for j in range(l3_size)])

        np.save("Weights/W1_offset_data.npy", W1_values_with_offset)
        np.save("Weights/W2_offset_data.npy", W2_values_with_offset)
        np.save("Weights/W3_offset_data.npy", W3_values_with_offset)
        np.save("Weights/b1_offset_data.npy", b1_values_with_offset)
        np.save("Weights/b2_offset_data.npy", b2_values_with_offset)
        np.save("Weights/b3_offset_data.npy", b3_values_with_offset)
        
        
        vw = VerifyWeights(n, l1, l2, flip_idxs, tol, W1_values, W2_values, W3_values, b1_values, b2_values, b3_values,
                    W1_values_with_offset, W2_values_with_offset, W3_values_with_offset,
                    b1_values_with_offset, b2_values_with_offset, b3_values_with_offset)
        vw.main(anyflip="_Any")

        if iter == 3:
            return
        else:
            trn = RunNN_preset(X, y_true, hs1=l1, hs2=l2, out_size=1, lr = 0.1, epoch=10000, preset_weights=True)
            nn, y_predict = trn.TrainReturnWeights()
            with open(accuracy_file, "a") as f:
                f.write(f"Iteration: {iter},  Accuracy: {np.mean(y_true == y_predict):.4f}\n")

            RunForward(nn, X, y, y_true, tol, n, 1, l1, l2, iter+1)

    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()