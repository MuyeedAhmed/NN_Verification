from z3 import *
import numpy as np

def relu(x):
    return If(x > 0, x, 0)

def forward_pass_with_constraints(X, y, W1, b1, W2, b2, W3, b3, solver):
    l1_size = len(W1[0])
    l2_size = len(W2[0])

    y_predict = [Real(f'y_pred_{j}') for j in range(len(y))] 
    Z3 = [Real(f'Z3_{j}') for j in range(len(y))] 

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        y_row = y[row_idx]

        Z1 = [
            sum([
                RealVal(X_row[i]) * W1[i][j]
                for i in range(len(X_row))
            ]) + b1[0][j]
            for j in range(len(W1[0]))
        ]
        A1 = [relu(Z1[j]) for j in range(len(Z1))]

        Z2 = [
            sum([
                A1[i] * W2[i][j] 
                for i in range(len(A1))
            ]) + b2[0][j]
            for j in range(len(W2[0]))
        ]
        A2 = [relu(Z2[j]) for j in range(len(Z2))]
        
        solver.add(Z3[row_idx] == sum([
                A2[i] * W3[i][0] 
                for i in range(len(A2))
            ]) + b3[0][0])
        # Z3 = [
        #     sum([
        #         A2[i] * W3[i][j] 
        #         for i in range(len(A2))
        #     ]) + b3[0][j]
        #     for j in range(len(W3[0])) 
        # ]
        
        # A3 = [sigmoid_approx(Z3[j]) for j in range(len(Z3))]
        solver.add(y_predict[row_idx] == If(Z3[row_idx] >= 0, 1, 0))

        # A3 = [Real(f'A3_{i}') for i in range(len(Z3))]
        
        # for j in range(len(Z3)):
        #     solver.add(A3[j] == 1 / (1 + 2.7**(-Z3[j])))
        # # y_predict[row_idx] = If(A3[0] >= 0.5, 1, 0)
        # solver.add(y_predict[row_idx] == If(A3[0] >= 0.5, 1, 0))


        # if row_idx == 9:
        #     solver.add(y_predict[row_idx] != y_row[0])
        # else:
        # solver.add(y_predict[row_idx] == y_row[0])

    return y_predict
