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

print("W1")
print(nn.W1)
print("W2")
print(nn.W2)
print("W3")
print(nn.W3)
print("b1")
print(nn.b1)
print("b2")
print(nn.b2)
print("b3")
print(nn.b3)


X_test = X[10:20]
y_test = predictions[10:20]
# print("Y true", y[10:20].reshape(1, -1))
# print("Y pred", y_test.reshape(1, -1))
# print("Y pred again", nn.predict(X_test).reshape(1, -1))

solver = Solver()

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
# W1_offset, W2_offset, y_predict = forward_pass_with_constraints(X_filtered[:10], y_filtered[:10], nn.W1, nn.b1, nn.W2, nn.b2, nn.W3, nn.b3, solver)

''' Flip any one'''
flip_idx = Int('flip_idx')
solver.add(flip_idx >= 0, flip_idx < len(X_test))

for row_idx in range(len(X_test)):
    solver.add(
        If(row_idx == flip_idx, y_predict[row_idx] != y_test[row_idx][0], 
            y_predict[row_idx] == y_test[row_idx][0])
    )


limitW1 = [[0 for j in range(l1_size)] for i in range(len(nn.W1))]
limitW2 = [[0 for j in range(l2_size)] for i in range(len(nn.W2))]
limitW3 = [[0 for j in range(l3_size)] for i in range(len(nn.W3))]
limitb1 = [0 for i in range(l1_size)]
limitb2 = [0 for i in range(l2_size)]
limitb3 = [0 for i in range(l3_size)]


f = open("Weights.txt", "w")

for i in range(20):
    print("------------------------------")
    f.write("------------------------------\n")
    print("Iter", i)
    f.write(f"Iter {i}\n")
    solver.push()
    if i != 0:
        for i in range(len(nn.W1)):
            for j in range(l1_size):
                bound = abs(limitW1[i][j])
                if bound == 0:
                    solver.add(And(W1_offset[i][j] == 0))
                    continue
                solver.add(And(W1_offset[i][j] > -bound, W1_offset[i][j] < bound))
        for i in range(len(nn.W2)):
            for j in range(l2_size):
                bound = abs(limitW2[i][j])
                if bound == 0:
                    solver.add(And(W2_offset[i][j] == 0))
                    continue
                solver.add(And(W2_offset[i][j] > -bound, W2_offset[i][j] < bound))
        for i in range(len(nn.W3)):
            for j in range(l3_size):
                bound = abs(limitW3[i][j])
                if bound == 0:
                    solver.add(And(W3_offset[i][j] == 0))
                    continue
                solver.add(And(W3_offset[i][j] > -bound, W3_offset[i][j] < bound))
        ''' bias '''
        for i in range(l1_size):
            bound = abs(limitb1[i])
            if bound == 0:
                solver.add(And(b1_offset[i] == 0))
                continue
            solver.add(And(b1_offset[i] > -bound, b1_offset[i] < bound))
        for i in range(l2_size):
            bound = abs(limitb2[i])
            if bound == 0:
                solver.add(And(b2_offset[i] == 0))
                continue
            solver.add(And(b2_offset[i] > -bound, b2_offset[i] < bound))
        for i in range(l3_size):
            bound = abs(limitb3[i])
            if bound == 0:
                solver.add(And(b3_offset[i] == 0))
                continue
            solver.add(And(b3_offset[i] > -bound, b3_offset[i] < bound))

    if solver.check() == sat:
        model = solver.model()
        print("Solution found:")
        f.write("Solution found:\n")
        # for i in range(len(W1_offset)):
        #     for j in range(len(W1_offset[0])):
        #         print(f"W1_offset[{i}][{j}] =", model[W1_offset[i][j]])

        # for i in range(len(W2_offset)):
        #     print("[", end="")
        #     for j in range(len(W2_offset[0])):
        #         print(model[W2_offset[i][j]], end=", ")
        #     print("]")
        print("Divide")
        print("W1")
        f.write("W1\n")
        for i in range(len(W1_offset)):
            print("\t[", end="")
            f.write("\t[")
            for j in range(len(W1_offset[0])):
                limitW1[i][j] = model[W1_offset[i][j]].as_fraction() / 4
                value = model[W1_offset[i][j]].as_fraction() / nn.W1[i][j]
                print(model[W1_offset[i][j]].as_fraction() + nn.W1[i][j], end=", ")
                # print(value, end=", ")
                f.write(f"{value}, ")
            print("]")
            f.write("]\n")

        print("W2")
        f.write("W2\n")
        for i in range(len(W2_offset)):
            print("\t[", end="")
            f.write("\t[")
            for j in range(len(W2_offset[0])):
                limitW2[i][j] = model[W2_offset[i][j]].as_fraction() / 4
                value = model[W2_offset[i][j]].as_fraction() / nn.W2[i][j]
                print(model[W2_offset[i][j]].as_fraction() + nn.W2[i][j], end=", ")
                # print(value, end=", ")
                f.write(f"{value}, ")
            print("]")
            f.write("]\n")

        print("W3")
        f.write("W3\n")
        for i in range(len(W3_offset)):
            print("\t[", end="")
            f.write("\t[")
            for j in range(len(W3_offset[0])):
                limitW3[i][j] = model[W3_offset[i][j]].as_fraction() / 4
                value = model[W3_offset[i][j]].as_fraction() / nn.W3[i][j]
                print(model[W3_offset[i][j]].as_fraction() + nn.W3[i][j], end=", ")
                # print(value, end=", ")
                f.write(f"{value}, ")
            print("]")
            f.write("]\n")
        
        print("b1: [", end='')
        for i in range(l1_size):
            limitb1[i] = model[b1_offset[i]].as_fraction()/2
            print(model[b1_offset[i]].as_fraction()+nn.b1[0, i], end=", ")
            # print(model[b1_offset[i]].as_fraction()/nn.b1[0, i], end=", ")
        print("]")
        print("b2: [", end='')
        for i in range(l2_size):
            limitb2[i] = model[b2_offset[i]].as_fraction()/2
            print(model[b2_offset[i]].as_fraction()+nn.b2[0, i], end=", ")
            # print(model[b2_offset[i]].as_fraction()/nn.b2[0, i], end=", ")
        print("]")
        print("b3: [", end='')
        for i in range(l3_size):
            limitb3[i] = model[b3_offset[i]].as_fraction()/2
            print(model[b3_offset[i]].as_fraction()+nn.b3[0, i], end=", ")
            # print(model[b3_offset[i]].as_fraction()/nn.b3[0, i], end=", ")
        print("]")
        
        print("y pred: [", end="")
        for i in range(len(y_test)):
            print(model.eval(y_predict[i]), end=" ")
        print("]")
        print("y true:", predictions[10:20].reshape(1,-1)[0]) 
    else:
        print("No solution found.")
        # for i in range(len(W1_offset)):
        #     for j in range(len(W1_offset[0])):
        #         limitW1[i][j] = model[W1_offset[i][j]].as_fraction()*1.5
        # for i in range(len(W2_offset)):
        #     for j in range(len(W2_offset[0])):
        #         limitW2[i][j] = model[W2_offset[i][j]].as_fraction()*1.5
        # for i in range(len(W3_offset)):
        #     for j in range(len(W3_offset[0])):
        #         limitW3[i][j] = model[W3_offset[i][j]].as_fraction()*1.5
    solver.pop()
f.close()
