import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from Network import RunNN

from FlipStability import RunForward_L2

input_folder = "../../Datasets"
output_file = "nn_config_results.csv"

layer_options = [2]
node_options = [3, 5]
activations = ["relu", "sigmoid"]
learning_rates = [0.01, 0.1]

columns = [
    "Dataset", "Shape", "Train Size", "Validation Size", "Test Size",
    "Hidden Layers", "Nodes per Layer", "Hidden Sizes",
    "Activation", "Learning Rate",
    "Train Accuracy", "Train F1 Score",
    "Val Accuracy", "Val F1 Score",
    "Test Accuracy", "Test F1 Score", 
    "Stability_MaxMagn", "Stability_MeanMagn", "Stability_SumMagn"
]

if os.path.exists(output_file):
    os.remove(output_file)

pd.DataFrame(columns=columns).to_csv(output_file, index=False)

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue
    if filename != "flare.csv" or filename != "PieChart3.csv" or filename != "fri_c1_1000_50.csv":
        continue


    path = os.path.join(input_folder, filename)
    data = pd.read_csv(path)

    if data.shape[0] > 1500:
        continue
    if data.shape[1] < 2:
        continue

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_rest, X_val, y_rest, y_val = train_test_split(X_scaled, y, test_size=50, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_rest, y_rest, test_size=0.3, random_state=42, stratify=y_rest)

    for num_layers in layer_options:
        for node_count in node_options:
            hidden_sizes = [node_count] * num_layers
            for activation in activations:
                for lr in learning_rates:
                    runner = RunNN(X_train, y_train, hidden_sizes=hidden_sizes, epoch=10000, activation=activation, lr=lr)
                    nn, train_preds = runner.TrainReturnWeights()
                    if np.isnan(nn.W[0]).any() or np.isnan(nn.b[0]).any() or np.isnan(nn.W[1]).any() or np.isnan(nn.b[1]).any():
                        print("NaN in weights, skipping...")
                        continue

                    val_preds = nn.predict(X_val)
                    test_preds = nn.predict(X_test)

                    max_abs_value, mean_value, sum_abs_value = RunForward_L2(nn, X_val, val_preds, activation, 1e-9, len(X_val), flipCount=1, l1=node_count, l2=node_count)

                    train_acc = accuracy_score(y_train, train_preds)
                    train_f1 = f1_score(y_train, train_preds)
                    print("Train Accuracy:", train_acc)
                    val_acc = accuracy_score(y_val, val_preds)
                    val_f1 = f1_score(y_val, val_preds)

                    test_acc = accuracy_score(y_test, test_preds)
                    test_f1 = f1_score(y_test, test_preds)

                    row = pd.DataFrame([[
                        filename,
                        f"{X.shape[0]}x{X.shape[1]}",
                        len(X_train),
                        len(X_val),
                        len(X_test),
                        num_layers,
                        node_count,
                        str(hidden_sizes),
                        activation,
                        lr,
                        train_acc,
                        train_f1,
                        val_acc,
                        val_f1,
                        test_acc,
                        test_f1, max_abs_value, mean_value, sum_abs_value
                    ]], columns=columns)

                    row.to_csv(output_file, mode="a", header=False, index=False)
                    print(f"Done: {filename} | Layers: {num_layers} | Nodes: {node_count} | Act: {activation} | LR: {lr}")
    continue
