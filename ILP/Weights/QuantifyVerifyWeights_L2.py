import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os
from scipy.stats import gmean

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, W1, W2, W3, b1, b2, b3):
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        return self.A3

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

class VerifyWeights:
    def __init__(self, n, l1, l2, flp_idx, tol, W1, W2, W3, b1, b2, b3, W1_with_offset, W2_with_offset, W3_with_offset, b1_with_offset, b2_with_offset, b3_with_offset):
        self.n = n
        self.l1 = l1
        self.l2 = l2
        self.flp_idx = flp_idx
        self.tol = tol
        
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        
        self.W1_with_offset = W1_with_offset
        self.W2_with_offset = W2_with_offset
        self.W3_with_offset = W3_with_offset
        self.b1_with_offset = b1_with_offset
        self.b2_with_offset = b2_with_offset
        self.b3_with_offset = b3_with_offset
        
    def LoadDataset(self, dataset_path=None):
        df = pd.read_csv("../Dataset/appendicitis.csv")
        X = df.iloc[:, :-1].to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

        self.X_test = X[0:self.n]
        self.y_test = y[0:self.n]

    def RunForward(self):
        nn_before = NeuralNetwork(self.W1, self.W2, self.W3, self.b1, self.b2, self.b3)
        self.labels_before = nn_before.predict(self.X_test).reshape(1, -1)[0]

        nn_after = NeuralNetwork(self.W1_with_offset, self.W2_with_offset, self.W3_with_offset, self.b1_with_offset, self.b2_with_offset, self.b3_with_offset)
        self.labels_after = nn_after.predict(self.X_test).reshape(1, -1)[0]
        missmatch = 0
        for i in range(len(self.labels_after)):
            if self.labels_after[i] != self.labels_before[i]:
                if i != self.flp_idx:
                    missmatch += 1
            else:
                if i == self.flp_idx:
                    missmatch += 1
        return missmatch

    def quantify_magnitude(self, missmatch, anyflip, write_in_csv = False):
        W1_magn = [[(self.W1_with_offset[i][j] - self.W1[i][j]) / (self.W1[i][j] + 10e-16)
                for j in range(self.W1.shape[1])] for i in range(self.W1.shape[0])]

        W2_magn = [[(self.W2_with_offset[i][j] - self.W2[i][j]) / (self.W2[i][j] + 10e-16)
                    for j in range(self.W2.shape[1])] for i in range(self.W2.shape[0])]

        W3_magn = [[(self.W3_with_offset[i][j] - self.W3[i][j]) / (self.W3[i][j] + 10e-16)
                    for j in range(self.W3.shape[1])] for i in range(self.W3.shape[0])]

        b1_magn = [(self.b1_with_offset[i] - self.b1[i]) / (self.b1[i] + 10e-16) for i in range(len(self.b1_with_offset))]
        b2_magn = [(self.b2_with_offset[i] - self.b2[i]) / (self.b2[i] + 10e-16) for i in range(len(self.b2_with_offset))]
        b3_magn = [(self.b3_with_offset[i] - self.b3[i]) / (self.b3[i] + 10e-16) for i in range(len(self.b3_with_offset))]

        W1_magn = np.array(W1_magn)
        W2_magn = np.array(W2_magn)
        W3_magn = np.array(W3_magn)
        b1_magn = np.array(b1_magn)
        b2_magn = np.array(b2_magn)
        b3_magn = np.array(b3_magn)
        all_magns = np.concatenate([W1_magn.flatten(), W2_magn.flatten(), W3_magn.flatten(),
                                    b1_magn.flatten(), b2_magn.flatten(), b3_magn.flatten()])

        max_abs_value = np.max(np.abs(all_magns))
        median_value = np.median(np.abs(all_magns))
        mean_value = np.mean(np.abs(all_magns))
        sum_abs_value = np.sum(np.abs(all_magns))
        geomean_value = gmean(np.abs(all_magns) + 1) - 1

        print("Mismatch:", missmatch)
        print("Sum of Absolute Magnitude:", sum_abs_value)
        print("Geometric Mean of Magnitude:", geomean_value)

        print("Max Absolute Magnitude:", max_abs_value)
        print("Median Magnitude:", median_value)
        print("Mean Magnitude:", mean_value)

        if write_in_csv:
            if not os.path.exists(f"Stats/Result_{self.l1}{self.l2}_{self.n}{anyflip}.csv"):
                with open(f"Stats/Result_{self.l1}{self.l2}_{self.n}{anyflip}.csv", "w") as f:
                    f.write("Test_Length,Threshold,Flip_ID,Mismatch,Max_Abs_magn,Median_magn,Mean_magn,Sum_Abs_magn,Geomean_magn\n")
            with open(f"Stats/Result_{self.l1}{self.l2}_{self.n}{anyflip}.csv", "a") as f:
                f.write(f"{self.n},{self.tol},{self.flp_idx},{missmatch},{max_abs_value},{median_value},{mean_value},{sum_abs_value},{geomean_value}\n")

    def save_log_in_file(self, anyflip):
        output_file = f"Outputs/Output_{self.l1}{self.l2}_{self.n}{anyflip}.txt"
        with open(output_file, "a") as f:
            f.write(f"----------{self.flp_idx}----------\n")

        with open(output_file, "a") as f:
            f.write("Before : ")
            for i in range(len(self.labels_before)):
                if i == self.flp_idx:
                    f.write(f"\033[0;32m{self.labels_before[i]}\033[0m ")
                else:
                    f.write(f"{self.labels_before[i]} ")
            f.write("\n")

        with open(output_file, "a") as f:
            f.write("After  : ")
            for i in range(len(self.labels_after)):
                if self.labels_after[i] != self.labels_before[i]:
                    f.write(f"\033[0;31m{self.labels_after[i]}\033[0m ")
                else:
                    f.write(f"{self.labels_after[i]} ")
            f.write("\n")

    
    def main(self, anyflip=""):
        self.LoadDataset()
        missmatch = self.RunForward()
        self.quantify_magnitude(missmatch, anyflip, True)
        self.save_log_in_file(anyflip)
