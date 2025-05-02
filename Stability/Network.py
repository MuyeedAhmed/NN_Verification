import numpy as np
from sklearn.preprocessing import StandardScaler

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class NN:
    def __init__(self, input_size, hidden_sizes, output_size, activation="relu", learning_rate=0.01):
        self.learning_rate = learning_rate
        self.activation_name = activation.lower()
        np.random.seed(0)

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01 for i in range(len(layer_sizes) - 1)]
        self.b = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

    def activate(self, x):
        if self.activation_name == "relu":
            return relu(x)
        elif self.activation_name == "sigmoid":
            return sigmoid(x)
        elif self.activation_name == "tanh":
            return tanh(x)

    def activate_derivative(self, x):
        if self.activation_name == "relu":
            return relu_derivative(x)
        elif self.activation_name == "sigmoid":
            return sigmoid_derivative(x)
        elif self.activation_name == "tanh":
            return tanh_derivative(x)

    def forward(self, X):
        self.Z = []
        self.A = [X]
        for i in range(len(self.W) - 1):
            z = self.A[-1] @ self.W[i] + self.b[i]
            self.Z.append(z)
            self.A.append(self.activate(z))
        z_out = self.A[-1] @ self.W[-1] + self.b[-1]
        self.Z.append(z_out)
        self.A.append(sigmoid(z_out))
        return self.A[-1]

    def backward(self, X, y, output):
        m = X.shape[0]
        dA = (output - y) / m
        dZ = [dA * sigmoid_derivative(self.Z[-1])]
        dW = [self.A[-2].T @ dZ[0]]
        db = [np.sum(dZ[0], axis=0, keepdims=True)]
        for i in range(len(self.W) - 2, -1, -1):
            dA = dZ[0] @ self.W[i + 1].T
            dZ.insert(0, dA * self.activate_derivative(self.Z[i]))
            dW.insert(0, self.A[i].T @ dZ[0])
            db.insert(0, np.sum(dZ[0], axis=0, keepdims=True))
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

class RunNN:
    def __init__(self, X, y, hidden_sizes=[2, 1], out_size=1, lr=0.1, epoch=1000, activation="relu"):
        self.X = X
        self.y = y
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size
        self.lr = lr
        self.epoch = epoch
        self.activation = activation

    def TrainReturnWeights(self):
        nn = NN(
            input_size=self.X.shape[1],
            hidden_sizes=self.hidden_sizes,
            output_size=self.out_size,
            activation=self.activation,
            learning_rate=self.lr
        )
        nn.train(self.X, self.y, epochs=self.epoch)
        predictions = nn.predict(self.X)
        return nn, predictions
