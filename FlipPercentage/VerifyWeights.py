import numpy as np
import pandas as pd
import sys
import os
from scipy.stats import gmean

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

class VerifyWeights:
    def __init__(self, X, y, n, l1, l2, activation, flp_idx, tol, W1, W2, W3, b1, b2, b3, 
                W1_with_offset, W2_with_offset, W3_with_offset, b1_with_offset, b2_with_offset, b3_with_offset, 
                y_gt=None, file_name=""):
        self.file_name = file_name
        self.X_test = X
        self.y_test = y
        if y_gt is not None:
            self.y_gt = y_gt
        self.n = n
        self.l1 = l1
        self.l2 = l2
        self.activation = activation
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

    def RunForward(self):
        nn_before = NeuralNetwork(self.W1, self.W2, self.W3, self.b1, self.b2, self.b3, self.activation)
        self.labels_before = nn_before.predict(self.X_test).reshape(1, -1)[0]
        
        nn_after = NeuralNetwork(self.W1_with_offset, self.W2_with_offset, self.W3_with_offset, self.b1_with_offset, self.b2_with_offset, self.b3_with_offset, self.activation)
        self.labels_after = nn_after.predict(self.X_test).reshape(1, -1)[0]

        return nn_before, nn_after

    def DetectMismatch(self):
        mismatch = 0
        for i in range(len(self.labels_after)):
            if self.labels_after[i] != self.labels_before[i]:
                mismatch += 1
        return mismatch

    # def DetectMismatch(self):
    #     mismatch = 0
    #     for i in range(len(self.labels_after)):
    #         if self.labels_after[i] != self.labels_before[i]:
    #             if i not in self.flp_idx:
    #                 mismatch += 1
    #         else:
    #             if i in self.flp_idx:
    #                 mismatch += 1
    #     return mismatch

    def quantify_magnitude(self, mismatch, Task, removed_feature = None, write_in_csv = False):
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

        print("Mismatch:", mismatch)
        print("Sum of Absolute Magnitude:", sum_abs_value)
        print("Geometric Mean of Magnitude:", geomean_value)

        print("Max Absolute Magnitude:", max_abs_value)
        print("Median Magnitude:", median_value)
        print("Mean Magnitude:", mean_value)

        if write_in_csv:
            if not os.path.exists(f"Stats/{Task}_{self.l1}{self.l2}.csv"):
                with open(f"Stats/{Task}_{self.l1}{self.l2}.csv", "w") as f:
                    if Task == "Flip_Any":
                        f.write("Dataset,N_samples,Threshold,Flip_Count,Mismatch,Max_Abs_magn,Median_magn,Mean_magn,Sum_Abs_magn,Geomean_magn\n")
                    elif Task == "Flip_Select":
                        f.write("Dataset,N_samples,Threshold,Flip_ID,Mismatch,Max_Abs_magn,Median_magn,Mean_magn,Sum_Abs_magn,Geomean_magn\n")
                    elif Task == "Feature_Selection":
                        f.write("Dataset,N_samples,Removed_Feature,Threshold,Flip_ID,Mismatch,Max_Abs_magn,Median_magn,Mean_magn,Sum_Abs_magn,Geomean_magn\n")
            
            with open(f"Stats/{Task}_{self.l1}{self.l2}.csv", "a") as f:
                if Task == "Flip_Any":
                    f.write(f"{self.file_name},{self.n},{self.tol},{len(self.flp_idx)},{mismatch},{max_abs_value},{median_value},{mean_value},{sum_abs_value},{geomean_value}\n")
                elif Task == "Flip_Select":
                    f.write(f"{self.file_name},{self.n},{self.tol},{self.flp_idx},{mismatch},{max_abs_value},{median_value},{mean_value},{sum_abs_value},{geomean_value}\n")
                elif Task == "Feature_Selection":
                    f.write(f"{self.file_name},{self.n},{removed_feature},{self.tol},{self.flp_idx},{mismatch},{max_abs_value},{median_value},{mean_value},{sum_abs_value},{geomean_value}\n")

    def save_log_in_file(self, Task, removed_feature=None):
        output_file = f"Outputs/{Task}_{self.l1}{self.l2}.txt"
        with open(output_file, "a") as f:
            f.write(f"----------{self.flp_idx}-----{self.file_name}-----\n")
            if removed_feature is not None:
                f.write(f"Removed Feature: {removed_feature}\n")
        with open(output_file, "a") as f:
            f.write("Before : ")
            for i in range(len(self.labels_before)):
                if i in self.flp_idx:
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

    

    def OutputDiffernce(self):
        diff_before = []
        for i in range(len(self.labels_before)):
            if self.labels_before[i] == 1:
                diff_before.append(nn_before.A3[i] - 0.5)
            else:
                diff_before.append(0.5 - nn_before.A3[i])

        diff_before = np.array(diff_before)
        print("Mean Difference Before:", np.mean(diff_before))


        diff_after = []
        for i in range(len(self.labels_after)):
            if self.labels_after[i] == 1:
                diff_after.append(nn_after.A3[i] - 0.5)
            else:
                diff_after.append(0.5 - nn_after.A3[i])

        diff_after = np.array(diff_after)
        print("Sum of Absolute Magnitude:", np.sum(np.abs(diff_after)))
        print("Mean Difference After:", np.mean(diff_after))

    def print_loss(self, nn_before, nn_after):
        loss_before = np.sum(np.abs(nn_before.A3 - self.y_test))
        loss_after = np.sum(np.abs(nn_after.A3 - self.y_test))
        print("Loss Before:", loss_before)
        print("Loss After:", loss_after)

    def main(self, Task, removed_feature=None):
        nn_before, nn_after = self.RunForward()
        mismatch = self.DetectMismatch()
        if Task == "BorderNetwork":
            self.print_loss(nn_before, nn_after)
        if Task == "MaximizeDifference":
            self.OutputDiffernce() # For Maximize Difference
        self.quantify_magnitude(mismatch, Task, removed_feature, write_in_csv=True)
        self.save_log_in_file(Task, removed_feature)

