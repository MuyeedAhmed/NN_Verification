import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os
from scipy.stats import gmean
from Network import NN, RunNN

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


def main():
    l1 = 4
    l2 = 4
    df = pd.read_csv("../Dataset/appendicitis.csv")
    X = df.iloc[:, :-1].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_true = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    trn = RunNN(X, y_true, hs1=l1, hs2=l2, out_size=1, lr = 0.1, epoch=10000)
    nn, y_predict = trn.TrainReturnWeights()
    

    l1_size = len(nn.W1[0])
    l2_size = len(nn.W2[0])
    l3_size = len(nn.W3[0])


    W1_values = np.array([[nn.W1[i][j] for j in range(l1_size)] for i in range(len(nn.W1))])
    W2_values = np.array([[nn.W2[i][j] for j in range(l2_size)] for i in range(len(nn.W2))])
    W3_values = np.array([[nn.W3[i][j] for j in range(l3_size)] for i in range(len(nn.W3))])
    b1_values = np.array([nn.b1[0, j] for j in range(l1_size)])
    b2_values = np.array([nn.b2[0, j] for j in range(l2_size)])
    b3_values = np.array([nn.b3[0, j] for j in range(l3_size)])
    times = 1.3
    W1_values_2 = W1_values*times
    W2_values_2 = W2_values*times
    W3_values_2 = W3_values*times
    b1_values_2 = b1_values*times
    b2_values_2 = b2_values*times
    b3_values_2 = b3_values*times

    nn_before = NeuralNetwork(W1_values, W2_values, W3_values, b1_values, b2_values, b3_values)
    labels_before = nn_before.predict(X).reshape(1, -1)[0]

    nn_after = NeuralNetwork(W1_values_2, W2_values_2, W3_values_2, b1_values_2, b2_values_2, b3_values_2) 
    labels_after = nn_after.predict(X).reshape(1, -1)[0]
    print(f"Labels before: {labels_before}")
    print(f"Labels after: {labels_after}")
    missmatch = 0
    for i in range(len(labels_after)):
        if labels_after[i] != labels_before[i]:
            missmatch += 1
    print(f"Out of {len(X)},  Missmatch: {missmatch}")
main()