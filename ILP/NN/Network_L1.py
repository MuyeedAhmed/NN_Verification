import numpy as np
from sklearn.preprocessing import StandardScaler


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NN:
    def __init__(self, input_size, hidden_size1, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        np.random.seed(0)
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2
    
    def backward(self, X, y, output):
        m = X.shape[0]

        dZ2 = (output - y) / m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
                # print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

class RunNN:
    def __init__(self, X, y, hs1=2, out_size=1, lr = 0.1, epoch=1000):
        self.X = X
        self.y = y
        self.hs1 = hs1
        self.out_size = out_size
        self.lr = lr
        self.epoch = epoch

    def TrainReturnWeights(self):
        nn = NN(input_size=self.X.shape[1], hidden_size1=self.hs1, output_size=self.out_size, learning_rate=self.lr)
        nn.train(self.X, self.y, epochs=self.epoch)

        predictions = nn.predict(self.X)
        return nn, predictions
