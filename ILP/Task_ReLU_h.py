from NN.Network_L1 import NN, RunNN
# from Gurobi_ForwardPass_L2_Indicator import ForwardPass
from Weights.QuantifyVerifyWeights_L1 import VerifyWeights

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import subprocess

timeLimit = 30

def main():
    n = 106
    l1 = 4
    flipCount = 1
    tol = 2e-9

    df = pd.read_csv("../Dataset/appendicitis.csv")
    X = df.iloc[:, :-1].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_true = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    trn = RunNN(X, y_true, hs1=l1, out_size=1, lr = 0.1, epoch=10000)
    nn, y_predict = trn.TrainReturnWeights()
    
    X = X[0:n]
    y = y_predict[0:n]

    RunForward(nn, X, y, tol, n, 1, l1)
    

def RunForward(nn, X, y, tol, n, flipCount, l1):

    l1_size = len(nn.W1[0])
    l2_size = len(nn.W2[0])

    model = gp.Model("Minimize_b")

    W1_offset = model.addVars(len(nn.W1), l1_size, vtype=GRB.CONTINUOUS, name="W1_offset")
    W2_offset = model.addVars(len(nn.W2), l2_size, vtype=GRB.CONTINUOUS, name="W2_offset")

    b1_offset = model.addVars(l1_size, vtype=GRB.CONTINUOUS, name="b1_offset")
    b2_offset = model.addVars(l2_size, vtype=GRB.CONTINUOUS, name="b2_offset")

    NewW1 = [[nn.W1[i][j] + W1_offset[i, j] for j in range(l1_size)] for i in range(len(nn.W1))]
    NewW2 = [[nn.W2[i][j] + W2_offset[i, j] for j in range(l2_size)] for i in range(len(nn.W2))]
    Newb1 = [[nn.b1[0, i] + b1_offset[i] for i in range(l1_size)]]
    Newb2 = [[nn.b2[0, i] + b2_offset[i] for i in range(l2_size)]]

    Z1 = model.addVars(len(X), l1_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z1")
    h1 = model.addVars(len(X), l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="h1")
    A1 = model.addVars(len(X), l1_size, lb=0, name="A1") 

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        for j in range(l1_size):
            model.addConstr(
                Z1[row_idx, j] == sum(X_row[i] * nn.W1[i][j] for i in range(len(X_row))) + nn.b1[0][j],
                f"Z1_def_{row_idx}_{j}"
            )

            model.addConstr(Z1[row_idx, j] + h1[row_idx, j] >= 0, f"ReLU_pos_{row_idx}_{j}")
            model.addConstr(h1[row_idx, j] >= 0, f"ReLU_h_nonneg_{row_idx}_{j}")
            model.addConstr(A1[row_idx, j] == Z1[row_idx, j] + h1[row_idx, j], f"A1_def_{row_idx}_{j}")



    Z2 = model.addVars(len(X), l2_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z2")

    for row_idx in range(len(X)):
        for j in range(l2_size):
            model.addConstr(
                Z2[row_idx, j] == sum(A1[row_idx, i] * nn.W2[i][j] for i in range(l1_size)) + nn.b2[0][j],
                f"Z2_def_{row_idx}_{j}"
            )
    
    y_g = [model.addVar(vtype=GRB.BINARY, name=f"y2_{i}") for i in range(len(X))]
    f = model.addVars(len(X), vtype=GRB.BINARY, name=f"flip_i") 
    
    M = 50
    model.addConstr(sum(f[i] for i in range(len(X))) == flipCount, "one_flip")

    E = tol
    for i in range(len(X)):
        y_scalar = int(y[i])
        model.addConstr(Z2[i, 0] >= E - M * (1 - y_g[i]), f"Z2_{i}_lower_bound")
        model.addConstr(Z2[i, 0] <= -E + M * y_g[i], f"Z2_{i}_upper_bound")

        model.addConstr(f[i] >= y_g[i] - y_scalar, f"flip_upper_{i}")
        model.addConstr(f[i] >= y_scalar - y_g[i], f"flip_lower_{i}")
        model.addConstr(f[i] <= y_g[i] + y_scalar, f"flip_cap_{i}")
        model.addConstr(f[i] <= 2 - y_g[i] - y_scalar, f"flip_cap_2_{i}")

                    
    abs_b1 = model.addVars(l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b1")
    abs_b2 = model.addVars(l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_b2")

    abs_W1 = model.addVars(len(nn.W1), l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W1")
    abs_W2 = model.addVars(len(nn.W2), l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_W2")

    for i in range(l1_size):
        model.addConstr(abs_b1[i] >= b1_offset[i])
        model.addConstr(abs_b1[i] >= -b1_offset[i])

    for i in range(l2_size):
        model.addConstr(abs_b2[i] >= b2_offset[i])
        model.addConstr(abs_b2[i] >= -b2_offset[i])

    for i in range(len(nn.W1)):
        for j in range(l1_size):
            model.addConstr(abs_W1[i, j] >= W1_offset[i, j])
            model.addConstr(abs_W1[i, j] >= -W1_offset[i, j])

    for i in range(len(nn.W2)):
        for j in range(l2_size):
            model.addConstr(abs_W2[i, j] >= W2_offset[i, j])
            model.addConstr(abs_W2[i, j] >= -W2_offset[i, j])


    objective = (
        gp.quicksum(abs_b1[i] for i in range(l1_size)) +
        gp.quicksum(abs_b2[i] for i in range(l2_size)) +
        gp.quicksum(abs_W1[i, j] for i in range(len(nn.W1)) for j in range(l1_size)) +
        gp.quicksum(abs_W2[i, j] for i in range(len(nn.W2)) for j in range(l2_size))+
        gp.quicksum(h1[i, j] for i in range(len(X)) for j in range(l1_size))
    )

    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam("FeasibilityTol", 1e-9)
    model.addConstr(objective >= 0, "NonNegativeObjective")
    model.setParam('TimeLimit', timeLimit)
    model.optimize()

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

        Z2_values = [[Z2[i, j].X for j in range(l2_size)] for i in range(len(X))]
        print("y_pred:", y.reshape(1,-1)[0])
        print("Z2:", Z2_values)
        
        W1_values = np.array([[nn.W1[i][j] for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_values = np.array([[nn.W2[i][j] for j in range(l2_size)] for i in range(len(nn.W2))])
        b1_values = np.array([nn.b1[0, j] for j in range(l1_size)])
        b2_values = np.array([nn.b2[0, j] for j in range(l2_size)])
        
        W1_values_with_offset = np.array([[nn.W1[i][j] + W1_offset[i, j].X for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_values_with_offset = np.array([[nn.W2[i][j] + W2_offset[i, j].X for j in range(l2_size)] for i in range(len(nn.W2))])
        b1_values_with_offset = np.array([nn.b1[0, j] + b1_offset[j].X for j in range(l1_size)])
        b2_values_with_offset = np.array([nn.b2[0, j] + b2_offset[j].X for j in range(l2_size)])


        W1_offset_array = np.array([[W1_offset[i, j].X for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_offset_array = np.array([[W2_offset[i, j].X for j in range(l2_size)] for i in range(len(nn.W2))])

        b1_offset_array = np.array([b1_offset[j].X for j in range(l1_size)])
        b2_offset_array = np.array([b2_offset[j].X for j in range(l2_size)])

        print("W1_offset:\n", W1_offset_array)
        print("W2_offset:\n", W2_offset_array)
        print("b1_offset:", b1_offset_array)
        print("b2_offset:", b2_offset_array)


        Z1_values = [[Z1[i, j].X for j in range(l1_size)] for i in range(len(X))]
        h1_values = [[h1[i, j].X for j in range(l1_size)] for i in range(len(X))]

        print("Summary of ReLU behavior:")
        for i in range(len(X)):
            for j in range(l1_size):
                z = Z1_values[i][j]
                h = h1_values[i][j]
                relu = z + h
                if z >= 0:
                    if h > 0:
                        print(f"Sample {i}, Node {j}: Z1={z:.6f} >= 0 → h1={h:.6f}, ReLU(Z1)={relu:.10f}")
                else:
                    if z+h < 0: 
                        print(f"Sample {i}, Node {j}: Z1={z:.6f} <  0 → h1={h:.6f}, Z1+h1={relu:.10f}")

        
        vw = VerifyWeights(n, l1, flip_idxs, tol, W1_values, W2_values, b1_values, b2_values,
                    W1_values_with_offset, W2_values_with_offset,
                    b1_values_with_offset, b2_values_with_offset)
        vw.main(anyflip="_Any")

    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()