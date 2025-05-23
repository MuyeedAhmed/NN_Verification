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

def train_model(X, y_gt, l1, l2, save_path=None, preset_weights_path=None, max_epochs=500):
    X_train, X_val, y_train, y_val = train_test_split(X, y_gt, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    input_dim = X.shape[1]
    model = BinaryClassifier(input_dim, l1, l2)

    if preset_weights_path:
        # print(f"Loading preset weights from {preset_weights_path}")
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

    torch.save(model.state_dict(), save_path)
    
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
        "train_preds": y_train_tensor.numpy().flatten(),
        "val_preds": y_val_tensor.numpy().flatten()
    }

    return model, final_metrics
    
    




if __name__ == "__main__":
    Test = "Test1_l44_Tr80_Val20"
    dataset_dir = "../../Dataset"
    # dataset_dir = "../Dataset"
    accuracy_file = f"Stats/{Test}.csv"
    if not os.path.exists(accuracy_file):
        with open(accuracy_file, "w") as f:
            f.write("Dataset,Row,Col,Type,Tr_Acc,Val_Acc,Tr_loss,Val_loss\n")
    for file_name in os.listdir(dataset_dir):
        if not file_name.endswith(".csv"):
            continue
        if os.path.exists(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}"):
            continue
        file_path = os.path.join(dataset_dir, file_name)
        df = pd.read_csv(file_path)

        if not (50 <= len(df) <= 400):
            continue
        print("File:", file_name)
        X = df.iloc[:, :-1]
        y_gt = df.iloc[:, -1]

        TrainA_Path = f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/model.pth"
        TrainB_Path = f"Weights/{Test}/TrainB/{file_name.split('.')[0]}/model.pth"
        if not os.path.exists(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}"):
            os.makedirs(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}")
        if not os.path.exists(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}"):
            os.makedirs(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}")
        try:
            model, final_metrics_A = train_model(X, y_gt, 4, 4, save_path=TrainA_Path, max_epochs=10000)
            model, final_metrics_B = train_model(X, y_gt, 4, 4, save_path=TrainB_Path, preset_weights_path=TrainA_Path, max_epochs=10000)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
        np.save(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/train_preds.npy", final_metrics_A['train_preds'])
        np.save(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}/train_preds.npy", final_metrics_B['train_preds'])
        np.save(f"Weights/{Test}/TrainA/{file_name.split('.')[0]}/val_preds.npy", final_metrics_A['val_preds'])
        np.save(f"Weights/{Test}/TrainB/{file_name.split('.')[0]}/val_preds.npy", final_metrics_B['val_preds'])
        
        
        with open(accuracy_file, "a") as f:
            f.write(f"{file_name},{len(X)},{X.shape[1]},TrainA,{final_metrics_A['train_accuracy']},{final_metrics_A['val_accuracy']},{final_metrics_A['train_loss']},{final_metrics_A['val_loss']}\n")
            f.write(f"{file_name},{len(X)},{X.shape[1]},TrainB,{final_metrics_B['train_accuracy']},{final_metrics_B['val_accuracy']},{final_metrics_B['train_loss']},{final_metrics_B['val_loss']}\n")
        


