from NN.Network import NN, RunNN
from Gurobi_ForwardPass_L2_ReLU import ForwardPass
from Weights.QuantifyVerifyWeights_L2 import VerifyWeights

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import subprocess

timeLimit = 3000
global y_true
def main():
    global y_true
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
    trn = RunNN(X, y_true, hs1=l1, hs2=l2, out_size=1, lr = 0.1, epoch=10000)
    nn, y_predict = trn.TrainReturnWeights()
    # ns = [50, 75, 100, 40, 60]
    # for n in ns:
    X = X[0:n]
    y = y_predict[0:n]

    RunForward(nn, X, y, tol, n, 1, l1, l2)
    
    # # tolerances = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
    # tolerances = [5e-9, 1e-8, 5e-8]
    # times = []
    # for tol in tolerances:
    #     # for flipCount in range(1, 11):
    #     t0 = time.time()
    #     RunForward(nn, X, y, tol, n, 1, l1, l2)
    #     times.append(time.time()-t0)

    # print("Times:", times)

def loss(output, y):
    loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
    return loss

def RunForward(nn, X, y, tol, n, flipCount, l1, l2):

    l1_size = len(nn.W1[0])
    l2_size = len(nn.W2[0])
    l3_size = len(nn.W3[0])

    model = gp.Model("Minimize_Offset")

    W1_offset = model.addVars(len(nn.W1), l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="W1_offset")
    W2_offset = model.addVars(len(nn.W2), l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="W2_offset")
    W3_offset = model.addVars(len(nn.W3), l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="W3_offset")

    b1_offset = model.addVars(l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="b1_offset")
    b2_offset = model.addVars(l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="b2_offset")
    b3_offset = model.addVars(l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="b3_offset")


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
        
                    
    objective = ( 
        gp.quicksum(b1_offset[i] for i in range(l1_size)) + 
        gp.quicksum(b2_offset[i] for i in range(l2_size)) + 
        gp.quicksum(b3_offset[i] for i in range(l3_size)) +
        gp.quicksum(W1_offset[i, j] for i in range(len(nn.W1)) for j in range(l1_size)) +
        gp.quicksum(W2_offset[i, j] for i in range(len(nn.W2)) for j in range(l2_size)) +
        gp.quicksum(W3_offset[i, j] for i in range(len(nn.W3)) for j in range(l3_size))
    )

    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam("FeasibilityTol", 1e-9)
    model.addConstr(objective >= 0, "NonNegativeObjective")
    
    model.setParam('TimeLimit', timeLimit)
    model.optimize()

    print("New loss vs Old loss")
    NewW1_value = np.array([[nn.W1[i][j] + W1_offset[i, j].X for j in range(l1_size)] for i in range(len(nn.W1))])
    NewW2_value = np.array([[nn.W2[i][j] + W2_offset[i, j].X for j in range(l2_size)] for i in range(len(nn.W2))])
    NewW3_value = np.array([[nn.W3[i][j] + W3_offset[i, j].X for j in range(l3_size)] for i in range(len(nn.W3))])
    Newb1_value = np.array([[nn.b1[0, i] + b1_offset[i].X for i in range(l1_size)]])
    Newb2_value = np.array([[nn.b2[0, i] + b2_offset[i].X for i in range(l2_size)]])
    Newb3_value = np.array([[nn.b3[0, i] + b3_offset[i].X for i in range(l3_size)]])
    old_output = nn.forward(X)
    new_nn = NN(input_size=X.shape[1], hidden_size1=4, hidden_size2=4, output_size=1, learning_rate=0.1)
    new_nn.W1 = NewW1_value
    new_nn.W2 = NewW2_value
    new_nn.W3 = NewW3_value
    new_nn.b1 = Newb1_value
    new_nn.b2 = Newb2_value
    new_nn.b3 = Newb3_value
    new_output = new_nn.forward(X)
    print(old_output)
    print(y_true)
    old_loss = loss(old_output, y_true)
    new_loss = loss(new_output, y_true)
    print("Old: ", old_loss)
    print("New: ", new_loss)
    
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
        
        W1_offset_array = np.array([[W1_offset[i, j].X for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_offset_array = np.array([[W2_offset[i, j].X for j in range(l2_size)] for i in range(len(nn.W2))])
        W3_offset_array = np.array([[W3_offset[i, j].X for j in range(l3_size)] for i in range(len(nn.W3))])

        b1_offset_array = np.array([b1_offset[j].X for j in range(l1_size)])
        b2_offset_array = np.array([b2_offset[j].X for j in range(l2_size)])
        b3_offset_array = np.array([b3_offset[j].X for j in range(l3_size)])

        print("W1_offset:\n", W1_offset_array)
        print("W2_offset:\n", W2_offset_array)
        print("W3_offset:\n", W3_offset_array)
        print("b1_offset:", b1_offset_array)
        print("b2_offset:", b2_offset_array)
        print("b3_offset:", b3_offset_array)
        # W1_values_magn = np.array([[W1_offset[i, j].X/nn.W1[i][j] for j in range(l1_size)] for i in range(len(nn.W1))])
        # W2_values_magn = np.array([[W2_offset[i, j].X/nn.W2[i][j] for j in range(l2_size)] for i in range(len(nn.W2))])
        # W3_values_magn = np.array([[W3_offset[i, j].X/nn.W3[i][j] for j in range(l3_size)] for i in range(len(nn.W3))])
        # b1_values_magn = np.array([b1_offset[j].X/nn.b1[0, j] for j in range(l1_size)])
        # b2_values_magn = np.array([b2_offset[j].X/nn.b2[0, j] for j in range(l2_size)])
        # b3_values_magn = np.array([b3_offset[j].X/nn.b3[0, j] for j in range(l3_size)])

        vw = VerifyWeights(n, l1, l2, flip_idxs, tol, W1_values, W2_values, W3_values, b1_values, b2_values, b3_values,
                    W1_values_with_offset, W2_values_with_offset, W3_values_with_offset,
                    b1_values_with_offset, b2_values_with_offset, b3_values_with_offset)
        vw.main(anyflip="_Any")

    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()