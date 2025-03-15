import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork_BeforeNoise:
    def __init__(self):
        self.W1 = np.load("Weights/W1_data.npy")
        self.W2 = np.load("Weights/W2_data.npy")
        self.W3 = np.load("Weights/W3_data.npy")
        self.b1 = np.load("Weights/b1_data.npy")
        self.b2 = np.load("Weights/b2_data.npy")
        self.b3 = np.load("Weights/b3_data.npy")
        
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        # print("Before Z3", self.Z3.reshape(1, -1))

        return self.A3

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)



class NeuralNetwork_AfterNoise:
    def __init__(self):
        self.W1 = np.load("Weights/W1_offset_data.npy")
        self.W2 = np.load("Weights/W2_offset_data.npy")
        self.W3 = np.load("Weights/W3_offset_data.npy")
        self.b1 = np.load("Weights/b1_offset_data.npy")
        self.b2 = np.load("Weights/b2_offset_data.npy")
        self.b3 = np.load("Weights/b3_offset_data.npy")

    
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        # print("After Z3", self.Z3.reshape(1, -1))
        return self.A3

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

def print_magnitude(nn_before, nn_after):
    print("W1")
    for i in range(nn_before.W1.shape[0]):
        print("[", end='')
        for j in range(nn_before.W1.shape[1]):
            print(f"{(nn_after.W1[i][j] - nn_before.W1[i][j]) / nn_before.W1[i][j]:.5f}", end=", ")
        print("]")

    print("W2")
    for i in range(nn_before.W2.shape[0]):
        print("[", end='')
        for j in range(nn_before.W2.shape[1]):
            print(f"{(nn_after.W2[i][j] - nn_before.W2[i][j]) / nn_before.W2[i][j]:.5f}", end=", ")
        print("]")

    print("W3")
    for i in range(nn_before.W3.shape[0]):
        print("[", end='')
        for j in range(nn_before.W3.shape[1]):
            print(f"{(nn_after.W3[i][j] - nn_before.W3[i][j]) / nn_before.W3[i][j]:.5f}", end=", ")
        print("]")

    print("b1")
    print("[", end='')
    for i in range(len(nn_after.b1)):
        print(f"{(nn_after.b1[i] - nn_before.b1[i]) / nn_before.b1[i]:.5f}", end=", ")
    print("]")

    print("b2")
    print("[", end='')
    for i in range(len(nn_after.b2)):
        print(f"{(nn_after.b2[i] - nn_before.b2[i]) / nn_before.b2[i]:.5f}", end=", ")
    print("]")

    print("b3")
    print("[", end='')
    for i in range(len(nn_after.b3)):
        print(f"{(nn_after.b3[i] - nn_before.b3[i]) / nn_before.b3[i]:.5f}", end=", ")
    print("]")

    W1_diff = [[(nn_after.W1[i][j] - nn_before.W1[i][j]) / nn_before.W1[i][j] 
            for j in range(nn_before.W1.shape[1])] for i in range(nn_before.W1.shape[0])]

    W2_diff = [[(nn_after.W2[i][j] - nn_before.W2[i][j]) / nn_before.W2[i][j] 
                for j in range(nn_before.W2.shape[1])] for i in range(nn_before.W2.shape[0])]

    W3_diff = [[(nn_after.W3[i][j] - nn_before.W3[i][j]) / nn_before.W3[i][j] 
                for j in range(nn_before.W3.shape[1])] for i in range(nn_before.W3.shape[0])]

    b1_diff = [(nn_after.b1[i] - nn_before.b1[i]) / nn_before.b1[i] for i in range(len(nn_after.b1))]
    b2_diff = [(nn_after.b2[i] - nn_before.b2[i]) / nn_before.b2[i] for i in range(len(nn_after.b2))]
    b3_diff = [(nn_after.b3[i] - nn_before.b3[i]) / nn_before.b3[i] for i in range(len(nn_after.b3))]

    W1_diff = np.array(W1_diff)
    W2_diff = np.array(W2_diff)
    W3_diff = np.array(W3_diff)
    b1_diff = np.array(b1_diff)
    b2_diff = np.array(b2_diff)
    b3_diff = np.array(b3_diff)

    all_diffs = np.concatenate([W1_diff.flatten(), W2_diff.flatten(), W3_diff.flatten(),
                                b1_diff.flatten(), b2_diff.flatten(), b3_diff.flatten()])

    max_abs_value = np.max(np.abs(all_diffs))
    median_value = np.median(all_diffs)
    mean_value = np.mean(all_diffs)

    print("Max Absolute Difference:", max_abs_value)
    print("Median Difference:", median_value)
    print("Mean Difference:", mean_value)

df = pd.read_csv("../Dataset/appendicitis.csv")
X = df.iloc[:, :-1].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df.iloc[:, -1].to_numpy().reshape(-1, 1)
X_test = X[15:23]
y_test = y[15:23]
# print(y_test.reshape(1, -1))

nn_before = NeuralNetwork_BeforeNoise()
print(nn_before.predict(X_test).reshape(1, -1))

nn_after = NeuralNetwork_AfterNoise()
print(nn_after.predict(X_test).reshape(1, -1))

# print("Magnitude")
# print_magnitude(nn_before, nn_after)

