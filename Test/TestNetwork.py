from NN.Network import NN, RunNN
from NN_Z3_ForwardPass import forward_pass_with_constraints, relu
from z3 import *
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler

# data = load_iris()
# data = load_breast_cancer()
# X = data.data
# X = (X - np.min(X)) / (np.max(X) - np.min(X))
# y = data.target.reshape(-1, 1)

# df = pd.read_csv("/Users/muyeedahmed/Desktop/Gitcode/AD_Attack/Dataset/appendicitis.csv")
df = pd.read_csv("Dataset/appendicitis.csv")
X = df.iloc[:, :-1].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

trn = RunNN(X, y, hs1=3, hs2=2, out_size=1, lr = 0.1, epoch=10000)
nn, predictions = trn.TrainReturnWeights()
# print("NN done", len(X_filtered), len(X_wrong))
# for i in range(len(predictions)):
for i in range(15,20):
    f = False
    if nn.Z3[i][0] >= 0:
        # print(nn.Z3[i][0] >= 0)
        f = True
    print(predictions[i][0], ":", f, nn.Z3[i][0])

# print("nn.Z3.reshape(1, -1)")
# print(nn.Z3.reshape(1, -1))
# print("W1")
# print(nn.W1)
# print("W2")
# print(nn.W2)
# print("W3")
# print(nn.W3)
# print("b1")
# print(nn.b1)
# print("b2")
# print(nn.b2)
# print("b3")
# print(nn.b3)


