import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# import matplotlib
# matplotlib.use('Agg')

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, l1, l2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, l1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(l2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

def F1Score():
    l1 = 4
    l2 = 4
    dataset_dir = "../../Dataset"
    Test = "FP_l44_F1_10Start"
    StatsFile = f"Stats/{Test}.csv"
    trainA_base = f"Weights/{Test}/TrainA"
    trainC_base = f"Weights/{Test}/TrainC"

    Stats = pd.read_csv(StatsFile)
    Stats["F1"] = None

    for idx, row in Stats.iterrows():
        dataset_file = row['Dataset']
        dataset_name = dataset_file.replace(".csv", "")
        restart = int(row['Seed']/10)
        trainA_dir = os.path.join(trainA_base, dataset_name)
        trainC_dir = os.path.join(trainC_base, dataset_name)
        modelA_path = os.path.join(trainA_dir, f"model_{restart}.pth")
        modelC_path = os.path.join(trainC_dir, f"model_{restart}.pth")
        predsA_path = os.path.join(trainA_dir, f"train_preds_{restart}.npy")
        
        if os.path.exists(modelC_path) == False:
            print("Model C not found for dataset:", dataset_name)
            continue

        df = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        X_full = df.iloc[:, :-1].values
        y_gt_full = df.iloc[:, -1].values

        X_train, X_val, y_train, y_val = train_test_split(X_full, y_gt_full, test_size=0.1, random_state=(int(row['Seed']/10)*42))
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        y_train_pred = np.round(np.load(predsA_path))

        Stats.at[idx, 'F1'] = f1_score(y_train, y_train_pred)
        
    Stats.to_csv(f"Stats/{Test}_SummaryWithF1.csv", index=False)

def Filter():
    # Test = "FP_l44_F1_10Start"
    # StatsFile = f"Stats/{Test}_SummaryWithF1.csv"
    # Stats = pd.read_csv(StatsFile)
    # Stats_Grouped = Stats.groupby('Dataset')

    # for name, group in Stats_Grouped:
    #     group['Min_Flip'] = group['Flip_Percentage'].min()
    #     group['Max_Flip'] = group['Flip_Percentage'].max()
    #     group['Median_Flip'] = group['Flip_Percentage'].median()
    #     group['Median'] = group['F1'].median()
    #     group['Max_F1'] = group['F1'].max()
    #     group['Min_F1'] = group['F1'].min()


    Test = "FP_l44_F1_10Start"
    StatsFile = f"Stats/{Test}_SummaryWithF1.csv"
    Stats = pd.read_csv(StatsFile)

    # Group by 'Dataset' and compute the desired statistics for Flip_Percentage and F1
    summary = Stats.groupby('Dataset').agg({
        'Flip_Percentage': ['min', 'max', 'median'],
        'F1': ['min', 'max', 'median']
    })

    # Flatten the multi-level column names
    summary.columns = ['Min_Flip', 'Max_Flip', 'Median_Flip', 'Min_F1', 'Max_F1', 'Median_F1']

    # Save the summary to a CSV
    summary.to_csv(f"Stats/{Test}_FullSummary.csv")



if __name__ == "__main__":
    # F1Score()
    Filter()
