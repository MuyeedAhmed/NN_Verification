from NN.Network import NN, RunNN
from Gurobi_ForwardPass_L2_Indicator import ForwardPass

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import subprocess

output_file = "Output.txt"

def main():
    df = pd.read_csv("../Dataset/appendicitis.csv")
    X = df.iloc[:, :-1].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_true = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    trn = RunNN(X, y_true, hs1=3, hs2=3, out_size=1, lr = 0.1, epoch=10000)
    nn, y_predict = trn.TrainReturnWeights()
    
    X = X[15:35]
    y = y_predict[15:35]
    # y = y_predict
    # print(y_true[15:25].reshape(1,-1))
    # print(y_test.reshape(1,-1))

    runtimes = [] 
    for idx in range(len(X)):
        if idx != 3:
            continue
        t0 = time.time()
        RunForward(nn, X, y, idx)
        runtimes.append(time.time()-t0)
        # break

    print(np.mean(runtimes), runtimes)

def RunForward(nn, X, y, flp_idx):

    l1_size = len(nn.W1[0])
    l2_size = len(nn.W2[0])
    l3_size = len(nn.W3[0])

    model = gp.Model("Minimize_b")
    # model.setParam("MIPFocus", 1)


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
    # y_g = {i: model.addVar(vtype=GRB.BINARY, name=f"y2_{i}") for i in range(len(X))}
    # f = {i: model.addVar(vtype=GRB.BINARY, name=f"flip_{i}") for i in range(len(X))}

    # M = 1e16
    # model.addConstr(sum(f[i] for i in range(len(X))) == 1, "one_flip")

    # for i in range(len(X)):
    #     y_scalar = int(y[i])
    #     model.addConstr(Z3[i, 0] >= -M * (1 - y_g[i]), f"Z3_{i}_lower_bound")
    #     model.addConstr(Z3[i, 0] <= -0.0000000000001 + M * y_g[i], f"Z3_{i}_upper_bound")

    #     model.addConstr(y_g[i] - y_scalar <= f[i], f"flip_upper_{i}")
    #     model.addConstr(y_scalar - y_g[i] <= f[i], f"flip_lower_{i}")
        
    for i in range(len(X)):
        if i == flp_idx:
            if y[i] == 0:
                model.addConstr(Z3[i, 0] >= 0, f"Z3_{i}_positive")
            else:
                model.addConstr(Z3[i, 0] <= -0.0000001, f"Z3_{i}_negative")
        else:
            if y[i] == 1:
                model.addConstr(Z3[i, 0] >= 0, f"Z3_{i}_positive")
            else:
                model.addConstr(Z3[i, 0] <= -0.0000001, f"Z3_{i}_negative")
                
    objective = (
        # gp.quicksum(Z2[i, 0] for i in range(len(X)) if y_predict[i] == 1) - 
        # gp.quicksum(Z2[i, 0] for i in range(len(X)) if y_predict[i] == 0) + 
        gp.quicksum(b1_offset[i] * b1_offset[i] for i in range(l1_size)) + 
        gp.quicksum(b2_offset[i] * b2_offset[i] for i in range(l2_size)) + 
        gp.quicksum(b3_offset[i] * b3_offset[i] for i in range(l3_size)) +
        gp.quicksum(W1_offset[i, j] * W1_offset[i, j] for i in range(len(nn.W1)) for j in range(l1_size)) +
        gp.quicksum(W2_offset[i, j] * W2_offset[i, j] for i in range(len(nn.W2)) for j in range(l2_size)) +
        gp.quicksum(W3_offset[i, j] * W3_offset[i, j] for i in range(len(nn.W3)) for j in range(l3_size))
    )


    model.setObjective(objective, GRB.MINIMIZE)

    model.addConstr(objective >= 0, "NonNegativeObjective")
    # model.setParam(GRB.Param.TimeLimit, 10)
    model.setParam('MIPGap', 0.5)
    model.setParam('TimeLimit', 60)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        W1_values = np.array([[nn.W1[i][j] for j in range(l1_size)] for i in range(len(nn.W1))])
        np.save("Weights/W1_data.npy", W1_values)

        W2_values = np.array([[nn.W2[i][j] for j in range(l2_size)] for i in range(len(nn.W2))])
        np.save("Weights/W2_data.npy", W2_values)

        W3_values = np.array([[nn.W3[i][j] for j in range(l3_size)] for i in range(len(nn.W3))])
        np.save("Weights/W3_data.npy", W3_values)

        b1_values = np.array([nn.b1[0, j] for j in range(l1_size)])
        np.save("Weights/b1_data.npy", b1_values)

        b2_values = np.array([nn.b2[0, j] for j in range(l2_size)])
        np.save("Weights/b2_data.npy", b2_values)

        b3_values = np.array([nn.b3[0, j] for j in range(l3_size)])
        np.save("Weights/b3_data.npy", b3_values)


        W1_values = np.array([[nn.W1[i][j] + W1_offset[i, j].X for j in range(l1_size)] for i in range(len(nn.W1))])
        np.save("Weights/W1_offset_data.npy", W1_values)

        W2_values = np.array([[nn.W2[i][j] + W2_offset[i, j].X for j in range(l2_size)] for i in range(len(nn.W2))])
        np.save("Weights/W2_offset_data.npy", W2_values)

        W3_values = np.array([[nn.W3[i][j] + W3_offset[i, j].X for j in range(l3_size)] for i in range(len(nn.W3))])
        np.save("Weights/W3_offset_data.npy", W3_values)

        b1_values = np.array([nn.b1[0, j] + b1_offset[j].X for j in range(l1_size)])
        np.save("Weights/b1_offset_data.npy", b1_values)

        b2_values = np.array([nn.b2[0, j] + b2_offset[j].X for j in range(l2_size)])
        np.save("Weights/b2_offset_data.npy", b2_values)

        b3_values = np.array([nn.b3[0, j] + b3_offset[j].X for j in range(l3_size)])
        np.save("Weights/b3_offset_data.npy", b3_values)

        
        with open(output_file, "a") as f:
            print(f"----------{flp_idx}----------", file=f)
        with open(output_file, "a") as f:
            subprocess.run(["python", "Weights/TestWeights.py"], stdout=f, stderr=f, text=True)

        # print("W1_offset:")
        # for i in range(len(nn.W1)):
        #     print("[", end='')
        #     for j in range(l1_size):
        #         print(f"{W1_offset[i, j].X}", end=', ' if j < l1_size - 1 else '')
        #     print("]")

        # print("\nW2_offset:")
        # for i in range(len(nn.W2)):
        #     print("[", end='')
        #     for j in range(l2_size):
        #         print(f"{W2_offset[i, j].X}", end=', ' if j < l2_size - 1 else '')
        #     print("]")

        # print("\nW3_offset:")
        # for i in range(len(nn.W3)):
        #     print("[", end='')
        #     for j in range(l3_size):
        #         print(f"{W3_offset[i, j].X}", end=', ' if j < l3_size - 1 else '')
        #     print("]")

        # print("\nb1_offset:")
        # print("[", end='')
        # for j in range(l1_size):
        #     print(f"{b1_offset[j].X}", end=', ' if j < l1_size - 1 else '')
        # print("]")

        # print("\nb2_offset:")
        # print("[", end='')
        # for j in range(l2_size):
        #     print(f"{b2_offset[j].X}", end=', ' if j < l2_size - 1 else '')
        # print("]")

        # print("\nb3_offset:")
        # print("[", end='')
        # for j in range(l3_size):
        #     print(f"{b3_offset[j].X}", end=', ' if j < l3_size - 1 else '')
        # print("]")
        # print()
        Z3_values = [[Z3[i, j].X for j in range(l3_size)] for i in range(len(X))]
        print(y.reshape(1,-1)[0])
        print("Z3:", Z3_values)
        # y_g_values = [int(y_g[i].X) for i in range(len(X))]
        # print("y_g values:", y_g_values)
        
    else:
        print("No feasible solution found.")


main()