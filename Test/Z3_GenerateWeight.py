from z3 import *
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NN import NN
from sklearn.metrics import accuracy_score
import time

def relu(x):
    return If(x > 0, x, 0)

def sigmoid_approx(x):
    return x / (1 + Abs(x))

def forward_pass_with_constraints(X, y, solver):
    l1_size = [10, len(X[0])]
    l2_size = [5, l1_size[0]]
    
    y_predict = [Real(f'y_pred_{j}') for j in range(len(y))] 

    W1 = [[Real(f'W1_{i}_{j}') for j in range(l1_size[0])] for i in range(l1_size[1])]
    W2 = [[Real(f'W2_{i}_{j}') for j in range(l2_size[0])] for i in range(l2_size[1])]

    b1 = [Real(f'b1_{j}') for j in range(l1_size[0])]
    b2 = [Real(f'b2_{j}') for j in range(l2_size[0])]
    
    for row_idx in range(len(X)):
        X_row = X[row_idx]
        y_row = y[row_idx]

        # Z1 = [Real(f'Z1_{row_idx}_{j}') for j in range(l1_size)]
        # Z1 = [Real(f'Z1_{row_idx}_{j}') for j in range(l1_size)]
        Z1 = [
            sum([
                RealVal(float(X_row[i])) * W1[i][j]
                for i in range(len(X_row))
            ]) + b1[j]
            for j in range(len(W1[0]))
        ]
        # Z1 = [sum([RealVal(float(X_row[i])) * NewW1[i][j] for i in range(len(X_row))]) + RealVal(float(b1[j])) for j in range(len(W1[0]))]
        # Z1 = [Sum([RealVal(X_row[i]) * NewW1[i][j] for i in range(len(X_row))]) + b1[j] for j in range(l1_size)]
        A1 = [relu(Z1[j]) for j in range(len(Z1))]
        Z2 = [
            sum([
                A1[i] * W2[i][j]
                for i in range(len(A1))
            ]) + b2[j]
            for j in range(len(W2[0]))
        ]
        # Z2 = [sum(A1[i] * NewW2[i][j] for i in range(len(A1))) + b2[j] for j in range(l2_size)]
        A2 = [relu(Z2[j]) for j in range(len(Z2))]

        Z3 = sum(A2)
        y_prob = sigmoid_approx(Z3)
        
        y_predict[row_idx] = If(y_prob >= 0.5, 1, 0)
        
        solver.add(y_predict[row_idx] == y_row[0])

    return W1, W2, b1, b2, y_predict


data = load_breast_cancer()        
X = data.data
y = data.target.reshape(-1, 1)


def write_to_file(content):
    with open("NN_Z3_Replace_Stats.txt", "a") as file:
        file.write(content + "\n")

for n_sample in range(10, 500, 10):
    print("N:", n_sample)
    write_to_file(f"N: {n_sample}")

    X_test = X[:n_sample]
    y_test = y[:n_sample]


    time0 = time.time()
    solver = Solver()
    # solver.push()
    W1, W2, b1, b2, y_predict = forward_pass_with_constraints(X_test, y_test, solver)

    if solver.check() == sat:
        model = solver.model()
        print("Solution found")
        write_to_file("Solution found")

        # for i in range(len(W1)):
        #     for j in range(len(W1[0])):
        #         print(f"W1[{i}][{j}] =", model[W1[i][j]])
        
        # for i in range(len(W2)):
        #     for j in range(len(W2[0])):
        #         print(f"W2[{i}][{j}] =", model[W2[i][j]])
        # print("y pred: ", end="")
        # for i in range(len(y_test)):
        #     print(model.eval(y_predict[i]), end=", ")
        # print() 
    else:
        print("No solution found.-------")
    # solver.pop()
    runtime = time.time()-time0
    print("Runtime", runtime)
    write_to_file(f"Runtime: {runtime}\n")

