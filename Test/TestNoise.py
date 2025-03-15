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
        self.W1 = np.array([
            [ 0.88748419, -0.51314979, -1.74279036],
            [ 0.9209507,   1.29974093,  1.0707348 ],
            [ 0.81562159, -0.07469451, -0.38766179],
            [ 1.30031863,  0.25365509, -0.7963261 ],
            [-0.30502048,  0.02307838,  1.74736147],
            [-1.08638674,  2.21341007, -0.71303069],
            [-1.35528664,  0.07327901,  0.37118595]
        ])

        self.W2 = np.array([
            [ 2.12130403e+00, -1.00378947e-03],
            [ 2.69500727e+00,  1.68464467e-02],
            [-2.32185432e+00, -5.19471522e-03]
        ])

        self.W3 = np.array([
            [-4.17704279],
            [ 0.01349032]
        ])

        self.b1 = np.array([[ 0.5410318,  -0.7995619,   0.38681452]])
        self.b2 = np.array([[ 0.78450924, -0.01503005]])
        self.b3 = np.array([[1.62484371]])
        
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        print("Before Z3", self.Z3.reshape(1, -1))

        return self.A3

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)



class NeuralNetwork_AfterNoise:
    def __init__(self):
        self.W1 = np.array([
                [0.88748419, -0.51314979, -1.7357431070523173],
                [0.9389336482517177, 1.3225897561431774, 1.0707348],
                [0.81562159, -0.07469451, -0.38630254066128117],
                [1.30031863, 0.25365509, -0.7636549737895557],
                [-0.30502048, 0.02307838, 1.777395972787804],
                [-1.080875542933697, 2.2204125026292947, -0.71303069],
                [-1.35528664, 0.07327901, 0.3773846665140065]
        ])
        self.W2 = np.array([
                [2.1221747553376695, -0.00100378947],
                [2.696865987612056, 0.0168464467],
                [-2.31594744269661, -0.00519471522]
        ])
        self.W3 = np.array([
                [-4.17704279],
                [0.01349032]
        ])
        self.b1 = np.array([[0.57671423787664, -0.7542244053914502, 0.38681452]])
        self.b2 = np.array([[0.8013232576737022, -0.01503005]])
        self.b3 = np.array([[1.62484371]])


    
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        print("After Z3", self.Z3.reshape(1, -1))
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
        print(f"{(nn_after.b1[i] - nn_before.b1[0, i]) / nn_before.b1[0, i]:.5f}", end=", ")
    print("]")

    print("b2")
    print("[", end='')
    for i in range(len(nn_after.b2)):
        print(f"{(nn_after.b2[i] - nn_before.b2[0, i]) / nn_before.b2[0, i]:.5f}", end=", ")
    print("]")

    print("b3")
    print("[", end='')
    for i in range(len(nn_after.b3)):
        print(f"{(nn_after.b3[i] - nn_before.b3[0, i]) / nn_before.b3[0, i]:.5f}", end=", ")
    print("]")


df = pd.read_csv("Dataset/appendicitis.csv")
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

