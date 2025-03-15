from NN.Network import NN, RunNN
from z3 import *
import numpy as np


from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import csv


def relu(x):
    return If(x > 0, x, 0)

def sigmoid_approx(x):
    return x / (1 + Abs(x))

def forward_pass_with_constraints(X, y, W1, b1, W2, b2, solver, fraction, flipI):
    l1_size = len(W1[0])
    l2_size = len(W2[0])
    # print("Sizes, ", l1_size, l2_size, len(W1), len(W2))
    y_predict = [Real(f'y_pred_{j}') for j in range(len(y))] 

    W1_offset = []
    # W1_offset = [[Real(f'W1_offset_{i}_{j}') for j in range(l1_size)] for i in range(len(W1))]
    W2_offset = [[Real(f'W2_offset_{i}_{j}') for j in range(l2_size)] for i in range(len(W2))]

    # for i in range(len(W1)):
    #     for j in range(l1_size):
    #         if j != 0:
    #            solver.add(W1_offset[i][j] == 0) 

    # for i in range(len(W1)):
    #     for j in range(l1_size):
    #         if W1[i][j] >= 0:
    #            solver.add(And(W1_offset[i][j] < W1[i][j]/fraction, W1_offset[i][j] > -W1[i][j]/fraction))
    #         else:
    #            solver.add(And(W1_offset[i][j] > W1[i][j]/fraction, W1_offset[i][j] < -W1[i][j]/fraction))

    for i in range(len(W2)):
        for j in range(l2_size):
            bound = abs(W2[i][j]/fraction)
            solver.add(And(W2_offset[i][j] > -bound, W2_offset[i][j] < bound))

            # if W2[i][j] >= 0:
            #    solver.add(And(W2_offset[i][j] < W2[i][j]/fraction, W2_offset[i][j] > -W2[i][j]/fraction))
            # else:
            #    solver.add(And(W2_offset[i][j] > W2[i][j]/fraction, W2_offset[i][j] < -W2[i][j]/fraction))


    # NewW1 = [[W1[i][j] + W1_offset[i][j] for j in range(l1_size)] for i in range(len(W1))]
    NewW2 = [[W2[i][j] + W2_offset[i][j] for j in range(l2_size)] for i in range(len(W2))]

    for row_idx in range(len(X)):
        X_row = X[row_idx]
        y_row = y[row_idx]

        # Z1 = [Real(f'Z1_{row_idx}_{j}') for j in range(l1_size)]
        # Z1 = [Real(f'Z1_{row_idx}_{j}') for j in range(l1_size)]
        Z1 = [
            sum([
                RealVal(float(X_row[i])) * W1[i][j] # NewW1
                for i in range(len(X_row))
            ]) + RealVal(float(b1[0, j]))
            for j in range(len(W1[0]))
        ]
        # Z1 = [sum([RealVal(float(X_row[i])) * NewW1[i][j] for i in range(len(X_row))]) + RealVal(float(b1[j])) for j in range(len(W1[0]))]
        # Z1 = [Sum([RealVal(X_row[i]) * NewW1[i][j] for i in range(len(X_row))]) + b1[j] for j in range(l1_size)]
        A1 = [relu(Z1[j]) for j in range(len(Z1))]
        Z2 = [
            sum([
                A1[i] * NewW2[i][j]
                for i in range(len(A1))
            ]) + RealVal(float(b2[0, j]))
            for j in range(len(W2[0]))
        ]
        # Z2 = [sum(A1[i] * NewW2[i][j] for i in range(len(A1))) + b2[j] for j in range(l2_size)]
        A2 = [relu(Z2[j]) for j in range(len(Z2))]

        Z3 = sum(A2)
        y_prob = sigmoid_approx(Z3)
        
        y_predict[row_idx] = If(y_prob >= 0.5, 1, 0)
        if row_idx == flipI:
            # print("yes")
            solver.add(y_predict[row_idx] != y_row[0])
        else:
            solver.add(y_predict[row_idx] == y_row[0])

    return W1_offset, W2_offset, y_predict

csv_filename = "iris_2.csv"
data = load_iris()
X = data.data
X = (X - np.min(X)) / (np.max(X) - np.min(X))

y = data.target.reshape(-1, 1)

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["N", "Layer1_Size", "Layer2_Size", "Fraction", "Solutions", "Avg time"])

    n_samples = [3,5,10]
    for n_sample in n_samples:
        for l1 in range(1,5):
            for l2 in range(1, l1+1):
                print("n_sample, l1, l2 : ", n_sample, l1, l2)
                trn = RunNN(X, y, hs1=l1, hs2=l2, out_size=1, lr = 0.01, epoch=1000)
                X_filtered, y_filtered, X_wrong, y_wrong, nn = trn.TrainReturnWeights()
                
                print("NN done")
                print(len(y_filtered), len(y_wrong))

                X_test = X[:n_sample]
                y_test = y[:n_sample]

                fractions = [10, 4, 2, 1]
                for fraction in fractions:
                    # print(f"\tfraction {fraction}")
                    solution_found = False
                    runtime = []
                    if n_sample < 10:
                        flipRange = n_sample
                    else:
                        flipRange = 10
                    for flipI in range(flipRange):
                        # print("Flip point", flipI)
                        time0 = time.time()
                        solver = Solver()
                        solver.push()
                        W1_offset, W2_offset, y_predict = forward_pass_with_constraints(X_test, y_test, nn.W1, nn.b1, nn.W2, nn.b2, solver, fraction, flipI)

                        if solver.check() == sat:
                            model = solver.model()
                            solution_found = True

                            # print(f"\t\tFlip point {flipI} : solution found")

                            # print("Solution found:")
                            # for i in range(len(W1_offset)):
                            #     for j in range(len(W1_offset[0])):
                            #         print(f"W1_offset[{i}][{j}] =", model[W1_offset[i][j]])
                            
                            # for i in range(len(W2_offset)):
                            #     for j in range(len(W2_offset[0])):
                            #         print(f"W2_offset[{i}][{j}] =", model[W2_offset[i][j]])
                            # print("y pred: ", end="")
                            # for i in range(len(y_test)):
                            #     print(model.eval(y_predict[i]), end=", ")
                            # print() 
                        # else:
                        #     print("No solution found.")
                        solver.pop()

                        runtime.append(time.time()-time0)
                        if solution_found:
                            break
                    # print(f"Avg time {np.mean(runtime)} Times: {runtime}")
                    # print(f"\t\tAvg time {np.mean(runtime)}")

                    writer.writerow([n_sample, l1, l2, fraction, "Yes" if solution_found else "No", np.mean(runtime)])
