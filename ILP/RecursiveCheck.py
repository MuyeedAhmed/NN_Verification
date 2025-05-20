from NN.Network_PresetWeights import NN_preset, RunNN_preset
from Gurobi_ForwardPass_L2_ReLU import ForwardPass
# from Weights.QuantifyVerifyWeights_L2 import VerifyWeights
from VerifyWeights import VerifyWeights

import gurobipy as gp
from gurobipy import GRB
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import subprocess

timeLimit = 300
accuracy_file = "accuracy.csv"

def main():
    l1 = 4
    l2 = 4
    flipCount = 1
    tol = 2e-6

    dataset_dir = "../../Dataset"
    if not os.path.exists(accuracy_file):
        with open(accuracy_file, "w") as f:
            f.write("Dataset,n,col_size,Iteration,Accuracy\n")


    for file_name in os.listdir(dataset_dir):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(dataset_dir, file_name)
        df = pd.read_csv(file_path)

        if not (50 <= len(df) <= 200):
            continue
        # if file_name != "dbworld-bodies-stemmed.csv":
        #     continue
        print(f"Running dataset: {file_name} with {len(df)} rows")

        X = df.iloc[:, :-1].to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y_gt = df.iloc[:, -1].to_numpy().reshape(-1, 1)

        trn = RunNN_preset(X, y_gt, hs1=l1, hs2=l2, out_size=1, lr=0.1, epoch=10000)
        nn, y_predict = trn.TrainReturnWeights()
        
        y = y_predict

        with open(accuracy_file, "a") as f:
            f.write(f"{file_name},{len(X)},{X.shape[1]},0,{np.mean(y_gt == y_predict):.4f}\n")
        try:
            RunForward(file_name, nn, X, y, y_gt, tol, len(X), 1, l1, l2, 1)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    

def RunForward(file_name, nn, X, y, y_gt, tol, n, flipCount, l1, l2, iter):

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

    # model.setParam("FeasibilityTol", 1e-9)
    model.addConstr(objective >= 0, "NonNegativeObjective")
    model.setParam('TimeLimit', timeLimit)
    model.optimize()

    if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
        if model.SolCount == 0:
            print("Timeout")
            return
        with open("Solved_Flip.txt", "a") as file:
            file.write(f"{file_name}-----\n")

        f_values = [f[i].X for i in range(len(X))]
        y_g_values = [y_g[i].X for i in range(len(X))]
        flip_idxs = [i for i, val in enumerate(f_values) if val == 1]

        if len(flip_idxs) != flipCount:
            print(f"Error: Expected {flipCount} value of 1 in f_values, but found {len(flip_idxs)}")
            return

        print("f values:", f_values)
        print("y_g values:", y_g_values)

        # Z3_values = [[Z3[i, j].X for j in range(l3_size)] for i in range(len(X))]
        # print("y_pred:", y.reshape(1,-1)[0])
        # print("Z3:", Z3_values)
        
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

        vw = VerifyWeights(X, y, n, l1, l2, "relu", flip_idxs, tol, W1_values, W2_values, W3_values, b1_values, b2_values, b3_values,
            W1_values_with_offset, W2_values_with_offset, W3_values_with_offset,
            b1_values_with_offset, b2_values_with_offset, b3_values_with_offset, y_gt=y_gt, file_name = file_name)
        vw.main(Task="Flip")
    
        if not os.path.exists(f"Weights/{file_name}"):
            save_weights(nn, file_name, 0)

        trn = RunNN_preset(X, y_gt, hs1=l1, hs2=l2, out_size=1, lr = 0.1, epoch=10000, preset_weights=True)
        nn, y_predict = trn.TrainReturnWeights()
        save_weights(nn, file_name, iter)
        with open(accuracy_file, "a") as f:
            f.write(f"{file_name},{len(X)},{X.shape[1]},{iter},{np.mean(y_gt == y_predict):.4f}\n")

        if iter == 3:
            return
        else:
            RunForward(file_name, nn, X, y, y_gt, tol, n, flipCount, l1, l2, iter+1)

    else:
        print("No feasible solution found.")


def save_weights(nn, file_name, iter):
    file_name = file_name.split(".")[0]
    if not os.path.exists(f"Weights/{file_name}"):
        os.makedirs(f"Weights/{file_name}")
    np.save(f"Weights/{file_name}/W1_{iter}.npy", nn.W1)
    np.save(f"Weights/{file_name}/W2_{iter}.npy", nn.W2)
    np.save(f"Weights/{file_name}/W3_{iter}.npy", nn.W3)
    np.save(f"Weights/{file_name}/b1_{iter}.npy", nn.b1)
    np.save(f"Weights/{file_name}/b2_{iter}.npy", nn.b2)
    np.save(f"Weights/{file_name}/b3_{iter}.npy", nn.b3)



if __name__ == "__main__":
    main()