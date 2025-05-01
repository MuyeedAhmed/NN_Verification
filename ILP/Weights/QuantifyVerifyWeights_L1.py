import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from scipy.stats import gmean

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, W1, W2, b1, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

class VerifyWeights:
    def __init__(self, n, l1, flp_idx, tol, W1, W2, b1, b2, W1_with_offset, W2_with_offset, b1_with_offset, b2_with_offset):
        self.n = n
        self.l1 = l1
        self.flp_idx = flp_idx
        self.tol = tol

        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

        self.W1_with_offset = W1_with_offset
        self.W2_with_offset = W2_with_offset
        self.b1_with_offset = b1_with_offset
        self.b2_with_offset = b2_with_offset

    def LoadDataset(self, dataset_path=None):
        df = pd.read_csv("../Dataset/appendicitis.csv")
        X = df.iloc[:, :-1].to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

        self.X_test = X[0:self.n]
        self.y_test = y[0:self.n]

    def RunForward(self):
        nn_before = NeuralNetwork(self.W1, self.W2, self.b1, self.b2)
        self.labels_before = nn_before.predict(self.X_test).reshape(1, -1)[0]

        nn_after = NeuralNetwork(self.W1_with_offset, self.W2_with_offset, self.b1_with_offset, self.b2_with_offset)
        self.labels_after = nn_after.predict(self.X_test).reshape(1, -1)[0]

        mismatch = 0
        for i in range(len(self.labels_after)):
            if self.labels_after[i] != self.labels_before[i]:
                if i not in self.flp_idx:
                    mismatch += 1
            else:
                if i in self.flp_idx:
                    mismatch += 1
        return mismatch

    def quantify_magnitude(self, mismatch, anyflip, write_in_csv=False):
        W1_magn = [[(self.W1_with_offset[i][j] - self.W1[i][j]) / (self.W1[i][j] + 10e-16)
                    for j in range(self.W1.shape[1])] for i in range(self.W1.shape[0])]
        W2_magn = [[(self.W2_with_offset[i][j] - self.W2[i][j]) / (self.W2[i][j] + 10e-16)
                    for j in range(self.W2.shape[1])] for i in range(self.W2.shape[0])]

        b1_magn = [(self.b1_with_offset[i] - self.b1[i]) / (self.b1[i] + 10e-16) for i in range(len(self.b1_with_offset))]
        b2_magn = [(self.b2_with_offset[i] - self.b2[i]) / (self.b2[i] + 10e-16) for i in range(len(self.b2_with_offset))]

        W1_magn = np.array(W1_magn)
        W2_magn = np.array(W2_magn)
        b1_magn = np.array(b1_magn)
        b2_magn = np.array(b2_magn)
        all_magns = np.concatenate([W1_magn.flatten(), W2_magn.flatten(), b1_magn.flatten(), b2_magn.flatten()])

        max_abs_value = np.max(np.abs(all_magns))
        median_value = np.median(np.abs(all_magns))
        mean_value = np.mean(np.abs(all_magns))
        sum_abs_value = np.sum(np.abs(all_magns))
        geomean_value = gmean(np.abs(all_magns) + 1) - 1

        print("Mismatch:", mismatch)
        print("Sum of Absolute Magnitude:", sum_abs_value)
        print("Geometric Mean of Magnitude:", geomean_value)
        print("Max Absolute Magnitude:", max_abs_value)
        print("Median Magnitude:", median_value)
        print("Mean Magnitude:", mean_value)

        if write_in_csv:
            filename = f"Stats/Result_L1_{self.l1}_{self.n}{anyflip}.csv"
            if not os.path.exists(filename):
                with open(filename, "w") as f:
                    f.write("Test_Length,Threshold,Flip_Count,Mismatch,Max_Abs_magn,Median_magn,Mean_magn,Sum_Abs_magn,Geomean_magn\n")
            with open(filename, "a") as f:
                f.write(f"{self.n},{self.tol},{len(self.flp_idx)},{mismatch},{max_abs_value},{median_value},{mean_value},{sum_abs_value},{geomean_value}\n")

    def save_log_in_file(self, anyflip):
        output_file = f"Outputs/Output_L1_{self.l1}_{self.n}{anyflip}.txt"
        with open(output_file, "a") as f:
            f.write(f"----------{self.flp_idx}----------\n")
            f.write("Before : ")
            for i in range(len(self.labels_before)):
                f.write(f"{self.labels_before[i]} ")
            f.write("\nAfter  : ")
            for i in range(len(self.labels_after)):
                f.write(f"{self.labels_after[i]} ")
            f.write("\n")

    def RunForward_MaximizeDiff(self):
        nn_before = NeuralNetwork(self.W1, self.W2, self.b1, self.b2)
        nn_after = NeuralNetwork(self.W1_with_offset, self.W2_with_offset, self.b1_with_offset, self.b2_with_offset)

        self.labels_before = nn_before.predict(self.X_test).reshape(1, -1)[0]
        self.labels_after = nn_after.predict(self.X_test).reshape(1, -1)[0]

        mismatch = np.sum(self.labels_before != self.labels_after)

        diff_before = [abs(0.5 - nn_before.A2[i]) for i in range(len(self.labels_before))]
        diff_after = [abs(0.5 - nn_after.A2[i]) for i in range(len(self.labels_after))]

        print("Mean Difference Before:", np.mean(diff_before))
        print("Mean Difference After:", np.mean(diff_after))
        print("Mismatch, Max Diff:", mismatch)

    def main(self, flipCount=1, anyflip=""):
        self.LoadDataset()
        mismatch = self.RunForward()
        self.RunForward_MaximizeDiff()
        self.quantify_magnitude(mismatch, anyflip, True)
        self.save_log_in_file(anyflip)
