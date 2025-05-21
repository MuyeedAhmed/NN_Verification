import pandas as pd
import os
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


dataset_dir = "../../Dataset"


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, W1, W2, W3, b1, b2, b3, activation):
        self.activation = activation
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        if self.activation == "sigmoid":
            self.A1 = sigmoid(self.Z1)
        elif self.activation == "relu":
            self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        if self.activation == "sigmoid":
            self.A2 = sigmoid(self.Z2)
        elif self.activation == "relu":
            self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        return self.A3

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

def getARI(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(file_path):
        return -2, "", -1, -1
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_gt = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    
    W1_0 = np.load(f"Weights/{file_name.split('.')[0]}/W1_0.npy")
    W2_0 = np.load(f"Weights/{file_name.split('.')[0]}/W2_0.npy")
    W3_0 = np.load(f"Weights/{file_name.split('.')[0]}/W3_0.npy")
    b1_0 = np.load(f"Weights/{file_name.split('.')[0]}/b1_0.npy")
    b2_0 = np.load(f"Weights/{file_name.split('.')[0]}/b2_0.npy")
    b3_0 = np.load(f"Weights/{file_name.split('.')[0]}/b3_0.npy")
    W1_1 = np.load(f"Weights/{file_name.split('.')[0]}/W1_1.npy")
    W2_1 = np.load(f"Weights/{file_name.split('.')[0]}/W2_1.npy")
    W3_1 = np.load(f"Weights/{file_name.split('.')[0]}/W3_1.npy")
    b1_1 = np.load(f"Weights/{file_name.split('.')[0]}/b1_1.npy")
    b2_1 = np.load(f"Weights/{file_name.split('.')[0]}/b2_1.npy")
    b3_1 = np.load(f"Weights/{file_name.split('.')[0]}/b3_1.npy")


    nn_0 = NeuralNetwork(W1_0, W2_0, W3_0, b1_0, b2_0, b3_0, "relu")
    nn_1 = NeuralNetwork(W1_1, W2_1, W3_1, b1_1, b2_1, b3_1, "relu")

    

    y_predict_0 = nn_0.predict(X).reshape(1, -1)[0]
    y_predict_1 = nn_1.predict(X).reshape(1, -1)[0]
    
    loss_0 = np.sum(np.abs(nn_0.A3 - y_gt))
    loss_1 = np.sum(np.abs(nn_1.A3 - y_gt))

    loss_delta = (
        'Increased' if loss_1 > loss_0 else
        'Decreased' if loss_1 < loss_0 else
        'Same'
    )

    ari = adjusted_rand_score(y_predict_1, y_predict_0)
    # print(ari, loss_delta, loss_0, loss_1)
    return ari, loss_delta, loss_0, loss_1
        


def compare_files(folder_path, prefixes, i1, i2):
    for prefix in prefixes:
        f1 = os.path.join(folder_path, f"{prefix}_{i1}.npy")
        f2 = os.path.join(folder_path, f"{prefix}_{i2}.npy")
        cont1 = np.load(f1)
        cont2 = np.load(f2)
        if not os.path.exists(f1) or not os.path.exists(f2):
            return "Path not found"
        if not np.array_equal(cont1, cont2):
            return "Different"
        break
    return "Same"

def generate_summary(input_csv, weights_base_path, output_csv):
    df = pd.read_csv(input_csv)
    grouped = df.groupby('Dataset')
    output_data = []

    for name, group in grouped:
        print(f"Processing {name}")
        if 0 not in group['Iteration'].values or 1 not in group['Iteration'].values:
            continue

        iter_group = group[group['Iteration'].isin([0, 1])]
        iter_group = iter_group.sort_values(by='Iteration')

        acc_0 = iter_group[iter_group['Iteration'] == 0]['Accuracy'].values[0]
        acc_1 = iter_group[iter_group['Iteration'] == 1]['Accuracy'].values[0]

        delta = (
            'Increased' if acc_1 > acc_0 else
            'Decreased' if acc_1 < acc_0 else
            'Same'
        )

        folder_name = name.replace('.csv', '')
        folder_path = os.path.join(weights_base_path, folder_name)

        weights_status = compare_files(folder_path, ['W1', 'W2', 'W3'], 0, 1)
        biases_status = compare_files(folder_path, ['b1', 'b2', 'b3'], 0, 1)

        ari, loss_delta, loss_0, loss_1 = getARI(name)

        output_data.append({
            'Dataset': name,
            'n': iter_group['n'].iloc[0],
            'col_size': iter_group['col_size'].iloc[0],
            'Accuracy_0': acc_0,
            'Accuracy_1': acc_1,
            'Delta_1': delta,
            'WeightChange_0_1': weights_status,
            'BiasChange_0_1': biases_status,
            'ARI': ari,
            'LossDelta': loss_delta,
            'Loss_0': loss_0,
            'Loss_1': loss_1
        })

    summary_df = pd.DataFrame(output_data)
    summary_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = "Stats/RecursiveCheck_Accuracy.csv"
    weights_base_path = "Weights"
    output_csv = "Stats/RecursiveCheck_Accuracy_Summary_ARI.csv"
    generate_summary(input_csv, weights_base_path, output_csv)

