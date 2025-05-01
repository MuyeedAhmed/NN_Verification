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

class NN_preset:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01, preset_weights=False):
        self.learning_rate = learning_rate
        if preset_weights:
            self.W1 = np.load("Weights/W1_offset_data.npy")
            self.W2 = np.load("Weights/W2_offset_data.npy")
            self.W3 = np.load("Weights/W3_offset_data.npy")
            self.b1 = [np.load("Weights/b1_offset_data.npy")]
            self.b2 = [np.load("Weights/b2_offset_data.npy")]
            self.b3 = [np.load("Weights/b3_offset_data.npy")]
        else:
            np.random.seed(0)
            self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
            self.b1 = np.zeros((1, hidden_size1))
            self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
            self.b2 = np.zeros((1, hidden_size2))
            self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
            self.b3 = np.zeros((1, output_size))

    
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)

        return self.A3
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        dZ3 = (output - y) / m
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
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
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
                # print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

class RunNN_preset:
    def __init__(self, X, y, hs1=2, hs2=1, out_size=1, lr = 0.1, epoch=1000, preset_weights=False):
        self.X = X
        self.y = y
        self.hs1 = hs1
        self.hs2 = hs2
        self.out_size = out_size
        self.lr = lr
        self.epoch = epoch
        self.preset_weights = preset_weights

    def TrainReturnWeights(self):
        nn = NN_preset(input_size=self.X.shape[1], hidden_size1=self.hs1, hidden_size2=self.hs2, output_size=self.out_size, learning_rate=self.lr, preset_weights=self.preset_weights)
        nn.train(self.X, self.y, epochs=self.epoch)

        predictions = nn.predict(self.X)
        # print("NN Pred", predictions.reshape(1, -1))

        # incorrect_indices = np.where(predictions.flatten() != self.y.flatten())[0]
        # correct_indices = np.where(predictions.flatten() == self.y.flatten())[0]

        # X_wrong = self.X[incorrect_indices]
        # y_wrong = self.y[incorrect_indices]

        # X_filtered = self.X[correct_indices]
        # y_filtered = self.y[correct_indices]
        
        # return nn, predictions, X_filtered, y_filtered, X_wrong, y_wrong
        return nn, predictions
