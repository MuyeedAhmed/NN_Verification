import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def relu(x):
    return np.maximum(0, x)

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

        return self.A3

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
    
df = pd.read_csv("../Dataset/appendicitis.csv")
X = df.iloc[:, :-1].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y_true = df.iloc[:, -1].to_numpy().reshape(-1, 1)
nn = NeuralNetwork_BeforeNoise()
y_predict = nn.predict(X)


X = X[15:23]
y = y_predict[15:23]
# print(y_true[15:25].reshape(1,-1))
# print(y_test.reshape(1,-1))


l1_size = len(nn.W1[0])
l2_size = len(nn.W2[0])
l3_size = len(nn.W3[0])


model = gp.Model("Minimize_b")

# model.setParam("MIPFocus", 1)

W1_offset = model.addVars(len(nn.W1), l1_size, vtype=GRB.CONTINUOUS, name="W1_offset")
W2_offset = model.addVars(len(nn.W2), l2_size, vtype=GRB.CONTINUOUS, name="W2_offset")
W3_offset = model.addVars(len(nn.W3), l3_size, vtype=GRB.CONTINUOUS, name="W3_offset")

b1_offset = model.addVars(l1_size, vtype=GRB.CONTINUOUS, name="b1_offset")
b2_offset = model.addVars(l2_size, vtype=GRB.CONTINUOUS, name="b2_offset")
b3_offset = model.addVars(l3_size, vtype=GRB.CONTINUOUS, name="b3_offset")

NewW1 = [[nn.W1[i][j] + W1_offset[i, j] for j in range(l1_size)] for i in range(len(nn.W1))]
NewW2 = [[nn.W2[i][j] + W2_offset[i, j] for j in range(l2_size)] for i in range(len(nn.W2))]
NewW3 = [[nn.W3[i][j] + W3_offset[i, j] for j in range(l3_size)] for i in range(len(nn.W3))]
Newb1 = [[nn.b1[0, i] + b1_offset[i] for i in range(l1_size)]]
Newb2 = [[nn.b2[0, i] + b2_offset[i] for i in range(l2_size)]]
Newb3 = [[nn.b3[0, i] + b3_offset[i] for i in range(l3_size)]]


Z1 = model.addVars(len(X), l1_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z1") 
A1 = model.addVars(len(X), l1_size, lb=0, name="A1") 

M = 100
z1 = model.addVars(len(X), l1_size, vtype=GRB.BINARY, name="z1")
z2 = model.addVars(len(X), l2_size, vtype=GRB.BINARY, name="z2")
z3 = model.addVars(len(X), l3_size, vtype=GRB.BINARY, name="z3")

for row_idx in range(len(X)):
    X_row = X[row_idx]
    for j in range(l1_size):
        model.addConstr(
            Z1[row_idx, j] == sum(X_row[i] * NewW1[i][j] for i in range(len(X_row))) + Newb1[0][j],
            f"Z1_def_{row_idx}_{j}"
        )

        model.addConstr(A1[row_idx, j] >= Z1[row_idx, j], f"ReLU_{row_idx}_{j}_pos")
        model.addConstr(A1[row_idx, j] >= 0, f"ReLU_{row_idx}_{j}_zero")
        model.addConstr(A1[row_idx, j] <= Z1[row_idx, j] + M * (1 - z1[row_idx, j]), f"ReLU_{row_idx}_{j}_BigM1")
        model.addConstr(A1[row_idx, j] <= M * z1[row_idx, j], f"ReLU_{row_idx}_{j}_BigM2")

Z2 = model.addVars(len(X), l2_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z2") 
A2 = model.addVars(len(X), l2_size, lb=0, name="A2") 

for row_idx in range(len(X)):
    for j in range(l2_size):
        model.addConstr(
            Z2[row_idx, j] == sum(A1[row_idx, i] * NewW2[i][j] for i in range(l1_size)) + Newb2[0][j],
            f"Z2_def_{row_idx}_{j}"
        )
        model.addConstr(A2[row_idx, j] >= Z2[row_idx, j], f"ReLU_A2_{row_idx}_{j}_pos")
        model.addConstr(A2[row_idx, j] >= 0, f"ReLU_A2_{row_idx}_{j}_zero")
        model.addConstr(A2[row_idx, j] <= Z2[row_idx, j] + M * (1 - z2[row_idx, j]), f"ReLU_A2_{row_idx}_{j}_BigM1")
        model.addConstr(A2[row_idx, j] <= M * z2[row_idx, j], f"ReLU_A2_{row_idx}_{j}_BigM2")

Z3 = model.addVars(len(X), l3_size,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z3")

for row_idx in range(len(X)):
    for j in range(l3_size):
        model.addConstr(
            Z3[row_idx, j] == sum(A2[row_idx, i] * NewW3[i][j] for i in range(l2_size)) + Newb3[0][j],
            f"Z3_def_{row_idx}_{j}"
        )


y_g = {i: model.addVar(vtype=GRB.BINARY, name=f"y2_{i}") for i in range(len(X))}
f = {i: model.addVar(vtype=GRB.BINARY, name=f"flip_{i}") for i in range(len(X))}

# M = 1e16
# model.addConstr(sum(f[i] for i in range(len(X))) == 1, "one_flip")

# for i in range(len(X)):
#     y_scalar = int(y[i])
#     model.addConstr(Z3[i, 0] >= -M * (1 - y_g[i]), f"Z3_{i}_lower_bound")
#     model.addConstr(Z3[i, 0] <= -0.0000000000001 + M * y_g[i], f"Z3_{i}_upper_bound")

#     model.addConstr(y_g[i] - y_scalar <= f[i], f"flip_upper_{i}")
#     model.addConstr(y_scalar - y_g[i] <= f[i], f"flip_lower_{i}")
    
for i in range(len(X)):
    # if i == 4:
    #     model.addConstr(Z3[i, 0] >= 0, f"Z3_{i}_positive")
    if i == 0:
        model.addConstr(Z3[i, 0] <= -0.00001, f"Z3_{i}_negative")
    else:
        if y[i] == 1:
            model.addConstr(Z3[i, 0] >= 0, f"Z3_{i}_positive")
        else:
            model.addConstr(Z3[i, 0] <= -0.00001, f"Z3_{i}_negative")
            
objective = (
    # gp.quicksum(Z2[i, 0] for i in range(len(X)) if y_predict[i] == 1) - 
    # gp.quicksum(Z2[i, 0] for i in range(len(X)) if y_predict[i] == 0) + 
    gp.quicksum(b1_offset[i] * b1_offset[i] for i in range(l1_size)) + 
    gp.quicksum(b2_offset[i] * b2_offset[i] for i in range(l2_size)) + 
    gp.quicksum(b3_offset[i] * b3_offset[i] for i in range(l3_size)) +
    gp.quicksum(W1_offset[i, j] * W1_offset[i, j] for i in range(len(nn.W1)) for j in range(l1_size)) +
    gp.quicksum(W2_offset[i, j] * W2_offset[i, j] for i in range(len(nn.W2)) for j in range(l2_size)) +
    gp.quicksum(W3_offset[i, j] * W3_offset[i, j] for i in range(len(nn.W3)) for j in range(l3_size))
)


model.setObjective(objective, GRB.MINIMIZE)

model.addConstr(objective >= 0, "NonNegativeObjective")

model.optimize()

if model.status == GRB.OPTIMAL:
    print("\t\tself.W1 = np.array([")
    for i in range(len(nn.W1)):
        print("\t\t\t[", end='')
        for j in range(l1_size):
            print(f"{nn.W1[i][j]+W1_offset[i, j].X}", end=', ' if j < l1_size - 1 else '')
        print("]", end=',\n' if i < len(nn.W1) - 1 else '\n')
    print("\t\t])")

    print("\t\tself.W2 = np.array([")
    for i in range(len(nn.W2)):
        print("\t\t\t[", end='')
        for j in range(l2_size):
            print(f"{nn.W2[i][j] + W2_offset[i, j].X}", end=', ' if j < l2_size - 1 else '')
        print("]", end=',\n' if i < len(nn.W2) - 1 else '\n')
    print("\t\t])")

    print("\t\tself.W3 = np.array([")
    for i in range(len(nn.W3)):
        print("\t\t\t[", end='')
        for j in range(l3_size):
            print(f"{nn.W3[i][j] + W3_offset[i, j].X}", end=', ' if j < l3_size - 1 else '')
        print("]", end=',\n' if i < len(nn.W3) - 1 else '\n')
    print("\t\t])")

    print("\t\tself.b1 = np.array([[", end='')
    for j in range(l1_size):
        print(f"{nn.b1[0, j] + b1_offset[j].X}", end=', ' if j < l1_size - 1 else '')
    print("]])")

    print("\t\tself.b2 = np.array([[", end='')
    for j in range(l2_size):
        print(f"{nn.b2[0, j] + b2_offset[j].X}", end=', ' if j < l2_size - 1 else '')
    print("]])")

    print("\t\tself.b3 = np.array([[", end='')
    for j in range(l3_size):
        print(f"{nn.b3[0, j] + b3_offset[j].X}", end=', ' if j < l3_size - 1 else '')
    print("]])")


    print("-------------------")
    print("W1_offset:")
    for i in range(len(nn.W1)):
        print("[", end='')
        for j in range(l1_size):
            print(f"{W1_offset[i, j].X}", end=', ' if j < l1_size - 1 else '')
        print("]")

    print("\nW2_offset:")
    for i in range(len(nn.W2)):
        print("[", end='')
        for j in range(l2_size):
            print(f"{W2_offset[i, j].X}", end=', ' if j < l2_size - 1 else '')
        print("]")

    print("\nW3_offset:")
    for i in range(len(nn.W3)):
        print("[", end='')
        for j in range(l3_size):
            print(f"{W3_offset[i, j].X}", end=', ' if j < l3_size - 1 else '')
        print("]")

    print("\nb1_offset:")
    print("[", end='')
    for j in range(l1_size):
        print(f"{b1_offset[j].X}", end=', ' if j < l1_size - 1 else '')
    print("]")

    print("\nb2_offset:")
    print("[", end='')
    for j in range(l2_size):
        print(f"{b2_offset[j].X}", end=', ' if j < l2_size - 1 else '')
    print("]")

    print("\nb3_offset:")
    print("[", end='')
    for j in range(l3_size):
        print(f"{b3_offset[j].X}", end=', ' if j < l3_size - 1 else '')
    print("]")
    print()
    Z3_values = [[Z3[i, j].X for j in range(l3_size)] for i in range(len(X))]
    print(y.reshape(1,-1)[0])
    print("Z3:", Z3_values)
    y_g_values = [int(y_g[i].X) for i in range(len(X))]
    print("y_g values:", y_g_values)
    # Z1_values = [
    #     [X[i, 0] * W1[0, j] + X[i, 1] * W1[1, j] + b1[j].X for j in range(2)]
    #     for i in range(10)
    # ]
    # print("Z1", Z1_values)

    # A1_values = [[A1[i, j].X for j in range(2)] for i in range(10)]
    # print("A1", A1_values)

    

    # Z2_values = [A1_values[i][0] * W2[0, 0] + A1_values[i][1] * W2[1, 0] + b2.X for i in range(10)]
    # print("Z2", Z2_values)
    
else:
    print("No feasible solution found.")
