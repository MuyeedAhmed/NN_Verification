# from NN.Network_PresetWeights import NN_preset, RunNN_preset
from Gurobi_ForwardPass_L2_ReLU import ForwardPass
# from VerifyWeights import VerifyWeights

import gurobipy as gp
from gurobipy import GRB
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import subprocess
import pandas as pd
import torch
from types import SimpleNamespace


from types import SimpleNamespace
import torch

Test = "Test1_l44_Tr80_Val20"
accuracy_file = f"Stats/{Test}.csv"
files_already_tested_path = "Stats/Tested.csv"


timeLimit = 1500


def extract_weights(load_path, transpose=True):
    state_dict = torch.load(load_path)
    nn = SimpleNamespace()

    W1 = state_dict['model.0.weight'].numpy()
    b1 = state_dict['model.0.bias'].numpy()
    W2 = state_dict['model.4.weight'].numpy()
    b2 = state_dict['model.4.bias'].numpy()
    W3 = state_dict['model.8.weight'].numpy()
    b3 = state_dict['model.8.bias'].numpy()

    if transpose:
        nn.W1 = W1.T
        nn.W2 = W2.T
        nn.W3 = W3.T
        nn.b1 = b1.reshape(1, -1)
        nn.b2 = b2.reshape(1, -1)
        nn.b3 = b3.reshape(1, 1)
    else:
        nn.W1 = W1    
        nn.W2 = W2
        nn.W3 = W3
        nn.b1 = b1
        nn.b2 = b2
        nn.b3 = b3

    return nn


def main():
    l1 = 4
    l2 = 4
    flipCount = 1
    tol = 2e-6

    dataset_dir = "../../Dataset"
    if not os.path.exists(accuracy_file):
        with open(accuracy_file, "w") as f:
            f.write("Dataset,n,col_size,Iteration,Accuracy\n")
    files_already_tested = pd.DataFrame(columns=["Dataset"])
    if not os.path.exists(files_already_tested_path):
        with open(files_already_tested_path, "w") as f:
            f.write("Dataset\n")
    else:
        files_already_tested = pd.read_csv(files_already_tested_path)

    for file_name in os.listdir(dataset_dir):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(dataset_dir, file_name)
        df = pd.read_csv(file_path)
        if not os.path.exists(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/train_preds.npy"):
            continue
        
        if os.path.exists(f"Weights/{Test}/TrainC/{file_name.split('.')[0]}/model.npy"):
            print("-----Already ran gurobi for this file:", file_name)
            continue

        if not (50 <= len(df) <= 400):
            continue
        print(f"----------------\nRunning dataset: {file_name} with {len(df)} rows\n----------------")
        
        with open(files_already_tested_path, "a") as file:
            file.write(f"{file_name}\n")
        
        X = df.iloc[:, :-1].to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y_gt = df.iloc[:, -1].to_numpy().reshape(-1, 1)
        X_train, X_val, y_train, y_val = train_test_split(X, y_gt, test_size=0.2, random_state=42)
        X = X_train
        y_gt = y_train
        try:
            y_train_pred = np.load(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/train_preds.npy")
            y_val_pred = np.load(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/val_preds.npy")
        
            nn = extract_weights(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/model.pth")
        except Exception as e:
            print(f"Error loading weights for {file_name}: {e}")
            continue

        try:
            RunForward(file_name, nn, X, y_train_pred, y_gt, tol, len(X), 1, l1, l2, 1)
            # if file_name in files_already_tested.values:
            #     print(f"Already tested: {file_name}")
            #     continue
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    

def RunForward(file_name, nn, X, y, y_gt, tol, n, flipCount, l1, l2, iter):
    y = y.reshape(-1, 1)
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
        with open("Stats/Solved_Flip.txt", "a") as file:
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

        original_model_path = f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/model.pth"
        original_state_dict = torch.load(original_model_path)  

        W1_torch = torch.tensor(W1_values_with_offset.T, dtype=torch.float32) 
        W2_torch = torch.tensor(W2_values_with_offset.T, dtype=torch.float32)
        W3_torch = torch.tensor(W3_values_with_offset.T, dtype=torch.float32)
        b1_torch = torch.tensor(b1_values_with_offset.flatten(), dtype=torch.float32)
        b2_torch = torch.tensor(b2_values_with_offset.flatten(), dtype=torch.float32)
        b3_torch = torch.tensor(b3_values_with_offset.flatten(), dtype=torch.float32)

        original_state_dict['model.0.weight'] = W1_torch
        original_state_dict['model.0.bias'] = b1_torch
        original_state_dict['model.4.weight'] = W2_torch
        original_state_dict['model.4.bias'] = b2_torch
        original_state_dict['model.8.weight'] = W3_torch
        original_state_dict['model.8.bias'] = b3_torch

        SavePath = f"Weights/{Test}/TrainE/{file_name.split('.')[0]}" # Change after testing for 1500 seconds
        if not os.path.exists(SavePath):
            os.makedirs(SavePath)
        torch.save(original_state_dict, f'{SavePath}/model.pth')
        

        # vw = VerifyWeights(X, y, n, l1, l2, "relu", flip_idxs, tol, W1_values, W2_values, W3_values, b1_values, b2_values, b3_values,
        #     W1_values_with_offset, W2_values_with_offset, W3_values_with_offset,
        #     b1_values_with_offset, b2_values_with_offset, b3_values_with_offset, y_gt=y_gt, file_name = file_name)
        # vw.main(Task="Flip")
    
        # if not os.path.exists(f"Weights/{file_name}"):
        save_weights(nn, file_name, 0)

    else:
        print("No feasible solution found.")



def save_weights(nn, file_name, iter):
    file_name = file_name.split(".")[0]
    if not os.path.exists(f"Weights/{file_name}"):
        os.makedirs(f"Weights/{file_name}")
    



if __name__ == "__main__":
    main()