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

df = pd.read_csv("/Users/muyeedahmed/Desktop/Gitcode/AD_Attack/Dataset/appendicitis.csv")
X = df.iloc[:, :-1].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

trn = RunNN(X, y, hs1=3, hs2=2, out_size=1, lr = 0.1, epoch=10000)
nn, predictions = trn.TrainReturnWeights()
# print(predictions.reshape(1, -1))
# print("NN done", len(X_filtered), len(X_wrong))

X_test = X[16:19]
y_test = predictions[16:19]

solver = Optimize()

l1_size = len(nn.W1[0])
l2_size = len(nn.W2[0])
l3_size = len(nn.W3[0])

W1_offset = [[Real(f'W1_offset_{i}_{j}') for j in range(l1_size)] for i in range(len(nn.W1))]
W2_offset = [[Real(f'W2_offset_{i}_{j}') for j in range(l2_size)] for i in range(len(nn.W2))]
W3_offset = [[Real(f'W3_offset_{i}_{j}') for j in range(l3_size)] for i in range(len(nn.W3))]
b1_offset = [Real(f'b1_offset_{i}') for i in range(l1_size)]
b2_offset = [Real(f'b2_offset_{i}') for i in range(l2_size)]
b3_offset = [Real(f'b3_offset_{i}') for i in range(l3_size)]

NewW1 = [[nn.W1[i][j] + W1_offset[i][j] for j in range(l1_size)] for i in range(len(nn.W1))]
NewW2 = [[nn.W2[i][j] + W2_offset[i][j] for j in range(l2_size)] for i in range(len(nn.W2))]
NewW3 = [[nn.W3[i][j] + W3_offset[i][j] for j in range(l3_size)] for i in range(len(nn.W3))]
Newb1 = [[nn.b1[0, i] + b1_offset[i] for i in range(l1_size)]]
Newb2 = [[nn.b2[0, i] + b2_offset[i] for i in range(l2_size)]]
Newb3 = [[nn.b3[0, i] + b3_offset[i] for i in range(l3_size)]]

y_predict = forward_pass_with_constraints(X_test, y_test, NewW1, Newb1, NewW2, Newb2, NewW3, Newb3, solver)

''' Flip any one'''
flip_idx = Int('flip_idx')
solver.add(flip_idx >= 0, flip_idx < len(X_test))

for row_idx in range(len(X_test)):
    solver.add(
        If(row_idx == flip_idx, y_predict[row_idx] != y_test[row_idx][0], 
                              y_predict[row_idx] == y_test[row_idx][0])
    )

''' Flip any two'''
# flip_idx1 = Int('flip_idx1')
# flip_idx2 = Int('flip_idx2')

# solver.add(flip_idx1 >= 0, flip_idx1 < len(X_test))
# solver.add(flip_idx2 >= 0, flip_idx2 < len(X_test))
# solver.add(flip_idx1 != flip_idx2)

# for row_idx in range(len(X_test)):
#     solver.add(
#         If(Or(row_idx == flip_idx1, row_idx == flip_idx2), 
#            y_predict[row_idx] != y_test[row_idx][0],
#            y_predict[row_idx] == y_test[row_idx][0])
#     )


for i in range(len(W1_offset)):
    for j in range(len(W1_offset[i])):
        solver.minimize(Abs(W1_offset[i][j]))

for i in range(len(W2_offset)):
    for j in range(len(W2_offset[i])):
        solver.minimize(Abs(W2_offset[i][j]))

for i in range(len(W3_offset)):
    for j in range(len(W3_offset[i])):
        solver.minimize(Abs(W3_offset[i][j]))

for i in range(len(b1_offset)):
    solver.minimize(Abs(b1_offset[i]))

for i in range(len(b2_offset)):
    solver.minimize(Abs(b2_offset[i]))

for i in range(len(b3_offset)):
    solver.minimize(Abs(b3_offset[i]))


if solver.check() == sat:
    model = solver.model()
    print("Solution found:")
    print("Divide")
    print("W1")
    for i in range(len(W1_offset)):
        print("\t[", end="")
        for j in range(len(W1_offset[0])):
            print(model[W1_offset[i][j]].as_fraction()/nn.W1[i][j], end=", ")
        print("]")
    print("W2")
    for i in range(len(W2_offset)):
        print("\t[", end="")
        for j in range(len(W2_offset[0])):
            print(model[W2_offset[i][j]].as_fraction()/nn.W2[i][j], end=", ")
        print("]")

    print("W3")
    for i in range(len(W3_offset)):
        print("\t[", end="")
        for j in range(len(W3_offset[0])):
            print(model[W3_offset[i][j]].as_fraction()/nn.W3[i][j], end=", ")
        print("]")
    
    print("y pred: [", end="")
    for i in range(len(y_predict)):
        print(model.eval(y_predict[i]), end=" ")
    print("]")
    print("y true:", predictions[16:19].reshape(1,-1)) 
else:
    print("No solution found.")

