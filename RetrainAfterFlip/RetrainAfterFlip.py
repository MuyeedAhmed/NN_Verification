import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import os
from types import SimpleNamespace
import gurobipy as gp
from gurobipy import GRB
from Gurobi_ForwardPass_L2_ReLU import ForwardPass


timeLimit = 300

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, l1, l2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, l1),
            nn.ReLU(),
            nn.BatchNorm1d(l1),
            nn.Dropout(0.3),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.BatchNorm1d(l2),
            nn.Dropout(0.3),
            nn.Linear(l2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_model(X, y_gt, l1, l2, val_size, save_path=None, preset_weights_path=None, max_epochs=500):
    X_train, X_val, y_train, y_val = train_test_split(X, y_gt, test_size=val_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    input_dim = X_train.shape[1]
    model = BinaryClassifier(input_dim, l1, l2)

    if preset_weights_path:
        model.load_state_dict(torch.load(preset_weights_path))

    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_loss = float('inf')
    patience = 15
    trigger_times = 0
    

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = criterion(val_preds, y_val_tensor).item()

        scheduler.step(val_loss)
        # print(f"[Epoch {epoch+1}] val_loss = {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_tensor).numpy().flatten()
        train_preds_tensor = torch.tensor(train_preds).unsqueeze(1)
        train_loss = criterion(train_preds_tensor, y_train_tensor).item()
        train_acc = accuracy_score(y_train_tensor.numpy().flatten(), np.round(train_preds))

        val_preds = model(X_val_tensor).numpy().flatten()
        val_preds_tensor = torch.tensor(val_preds).unsqueeze(1)
        val_loss = criterion(val_preds_tensor, y_val_tensor).item()
        val_acc = accuracy_score(y_val_tensor.numpy().flatten(), np.round(val_preds))

    final_metrics = {
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "train_preds": train_preds,
        "train_labels": y_train_tensor.numpy().flatten(),
        "val_preds": val_preds
    }

    torch.save(model.state_dict(), save_path)
    return model, final_metrics
    
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

def save_weights(nn, file_name):
    file_name = file_name.split(".")[0]
    if not os.path.exists(f"Weights/{file_name}"):
        os.makedirs(f"Weights/{file_name}")


def RunForward(Test, file_name, nn, X, y, y_gt, tol, n, flipCount, l1, l2):
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
            return False
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

        SavePath = f"Weights/{Test}/TrainC/{file_name.split('.')[0]}"
        if not os.path.exists(SavePath):
            os.makedirs(SavePath)
        torch.save(original_state_dict, f'{SavePath}/model.pth')
        

        # vw = VerifyWeights(X, y, n, l1, l2, "relu", flip_idxs, tol, W1_values, W2_values, W3_values, b1_values, b2_values, b3_values,
        #     W1_values_with_offset, W2_values_with_offset, W3_values_with_offset,
        #     b1_values_with_offset, b2_values_with_offset, b3_values_with_offset, y_gt=y_gt, file_name = file_name)
        # vw.main(Task="Flip")
    
        return True
    else:
        print("No feasible solution found.")
        return False



if __name__ == "__main__":
    l1 = 4
    l2 = 3
    val_size = 0.2
    epoch_count = 50000
    flipCount = 1
    tol = 3e-6

    Test = f"Test1_l{l1}{l2}_Variable"
    
    dataset_dir = "../../Dataset"
    # dataset_dir = "../Dataset"
    accuracy_file = f"Stats/{Test}.csv"
    error_file = f"Stats/Error_{Test}.txt"

    files_already_tested = pd.DataFrame(columns=["Dataset"])
    if not os.path.exists(accuracy_file):
        with open(accuracy_file, "w") as f:
            f.write("Dataset,Row,Col,Val_Size,Type,Tr_Acc,Val_Acc,Tr_loss,Val_loss\n")
    else:
        files_already_tested = pd.read_csv(accuracy_file)['Dataset'].unique()

    for file_name in os.listdir(dataset_dir):
        val_size = 0.2
        if not file_name.endswith(".csv"):
            continue
        if file_name in files_already_tested:
            print(f"Skipping {file_name} as it has already been processed.")
            continue
        
        file_path = os.path.join(dataset_dir, file_name)
        df = pd.read_csv(file_path)

        if not (100 <= len(df) <= 400):
            continue
        print("File:", file_name)
        X = df.iloc[:, :-1]
        y_gt = df.iloc[:, -1]
    
        # X_np = X.to_numpy()
        # scaler = StandardScaler()
        # X_np = scaler.fit_transform(X_np)
        # y_gt_np = y_gt.to_numpy().reshape(-1, 1)
        
        TrainA_Path = f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/model.pth"
        TrainB_Path = f"Weights/{Test}/TrainB/{file_name.split('.')[0]}/model.pth"
        if not os.path.exists(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}"):
            os.makedirs(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}")
        if not os.path.exists(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}"):
            os.makedirs(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}")
        try:
            while True:
                '''Step 1: Train A and B'''
                try:
                    model, final_metrics_A = train_model(X, y_gt, l1, l2, val_size, save_path=TrainA_Path, max_epochs=epoch_count)
                    model, final_metrics_B = train_model(X, y_gt, l1, l2, val_size, save_path=TrainB_Path, preset_weights_path=TrainA_Path, max_epochs=epoch_count*2)
                
                    np.save(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/train_preds.npy", final_metrics_A['train_preds'])
                    np.save(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}/train_preds.npy", final_metrics_B['train_preds'])
                    np.save(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/val_preds.npy", final_metrics_A['val_preds'])
                    np.save(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}/val_preds.npy", final_metrics_B['val_preds'])
                    
                    with open(accuracy_file, "a") as f:
                        f.write(f"{file_name},{len(X)},{X.shape[1]},{val_size},TrainA,{final_metrics_A['train_accuracy']},{final_metrics_A['val_accuracy']},{final_metrics_A['train_loss']},{final_metrics_A['val_loss']}\n")
                        f.write(f"{file_name},{len(X)},{X.shape[1]},{val_size},TrainB,{final_metrics_B['train_accuracy']},{final_metrics_B['val_accuracy']},{final_metrics_B['train_loss']},{final_metrics_B['val_loss']}\n")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    with open(error_file, "a") as f:
                        f.write(f"------------\n{file_name}\nStep 1: {e}\n---------------\n")
                    break
                
                '''Step 2: Run Gurobi Flip'''        
                try:
                    X_train, _, y_train, _ = train_test_split(X, y_gt, test_size=val_size, random_state=42)
                    y_train_pred = np.round(np.load(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/train_preds.npy"))
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)

                    extracted_weights = extract_weights(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/model.pth")
                    solution_found = RunForward(Test, file_name, extracted_weights, X_train, y_train_pred, y_train, tol, len(X), 1, l1, l2)
                except Exception as e:
                    with open(error_file, "a") as f:
                        f.write(f"------------\n{file_name}\nStep 1: {e}\n---------------\n")
                    break
                if solution_found:
                    '''Step 3: Retrain with the new weights'''
                    TrainC_Path = f"Weights/{Test}/TrainC/{file_name.split('.')[0]}/model.pth"
                    TrainD_Path = f"Weights/{Test}/TrainD/{file_name.split('.')[0]}/model.pth"
                    if not os.path.exists(f"Weights/{Test}/TrainD/{file_name.split('.')[0]}"):
                        os.makedirs(f"Weights/{Test}/TrainD/{file_name.split('.')[0]}")
                    model, final_metrics_D = train_model(X, y_gt, l1, l2, val_size, save_path=TrainD_Path, preset_weights_path=TrainC_Path, max_epochs=epoch_count)

                    
                    np.save(f"Weights/{Test}/TrainD/{file_name.split('.')[0]}/train_preds.npy", final_metrics_D['train_preds'])
                    np.save(f"Weights/{Test}/TrainD/{file_name.split('.')[0]}/val_preds.npy", final_metrics_D['val_preds'])
                    
                    with open(accuracy_file, "a") as f:
                        f.write(f"{file_name},{len(X)},{X.shape[1]},{val_size},TrainD,{final_metrics_D['train_accuracy']},{final_metrics_D['val_accuracy']},{final_metrics_D['train_loss']},{final_metrics_D['val_loss']}\n")


                    break
                else:
                    val_size += 0.1
                    if val_size > 0.5:
                        break
                    print("Increasing validation size to:", val_size)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            with open(error_file, "a") as f:
                f.write(f"------------\n{file_name}\n{e}\n---------------\n")
            continue

        


