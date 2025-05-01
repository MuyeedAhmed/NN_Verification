from NN.Network_Sigmoid import NN, RunNN
from Gurobi_ForwardPass_L2_Sigmoid import ForwardPass
from Weights.QuantifyVerifyWeights_L2_Sigmoid import VerifyWeights

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import subprocess

timeLimit = 6000

def main():
    n = 106
    l1 = 4
    l2 = 4

    df = pd.read_csv("../Dataset/appendicitis.csv")
    X = df.iloc[:, :-1].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_true = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    trn = RunNN(X, y_true, hs1=l1, hs2=l2, out_size=1, lr = 0.1, epoch=10000)
    nn, y_predict = trn.TrainReturnWeights()
    
    X = X[0:n]
    y = y_predict[0:n]

    tol = 2e-6
    RunForward(nn, X, y, -1, tol, n, l1, l2)



def RunForward(nn, X, y, flp_idx, tol, n, l1, l2):

    l1_size = len(nn.W1[0])
    l2_size = len(nn.W2[0])
    l3_size = len(nn.W3[0])

    model = gp.Model("Minimize_b")

    W1_offset = model.addVars(len(nn.W1), l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="W1_offset")
    W2_offset = model.addVars(len(nn.W2), l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="W2_offset")
    W3_offset = model.addVars(len(nn.W3), l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="W3_offset")

    b1_offset = model.addVars(l1_size, lb=0.0, vtype=GRB.CONTINUOUS, name="b1_offset")
    b2_offset = model.addVars(l2_size, lb=0.0, vtype=GRB.CONTINUOUS, name="b2_offset")
    b3_offset = model.addVars(l3_size, lb=0.0, vtype=GRB.CONTINUOUS, name="b3_offset")

    NewW1 = [[nn.W1[i][j] + W1_offset[i, j] for j in range(l1_size)] for i in range(len(nn.W1))]
    NewW2 = [[nn.W2[i][j] + W2_offset[i, j] for j in range(l2_size)] for i in range(len(nn.W2))]
    NewW3 = [[nn.W3[i][j] + W3_offset[i, j] for j in range(l3_size)] for i in range(len(nn.W3))]
    Newb1 = [[nn.b1[0, i] + b1_offset[i] for i in range(l1_size)]]
    Newb2 = [[nn.b2[0, i] + b2_offset[i] for i in range(l2_size)]]
    Newb3 = [[nn.b3[0, i] + b3_offset[i] for i in range(l3_size)]]

    Z3 = ForwardPass(model, X, NewW1, NewW2, NewW3, Newb1, Newb2, Newb3)
        
    for i in range(len(X)):
        if y[i] == 1:
            model.addConstr(Z3[i, 0] >= tol, f"Z3_{i}_positive")
        else:
            model.addConstr(Z3[i, 0] <= -tol, f"Z3_{i}_negative")


    
    objective = ( 
        gp.quicksum(b1_offset[i] for i in range(l1_size)) + 
        gp.quicksum(b2_offset[i] for i in range(l2_size)) + 
        gp.quicksum(b3_offset[i] for i in range(l3_size)) +
        gp.quicksum(W1_offset[i, j] for i in range(len(nn.W1)) for j in range(l1_size)) +
        gp.quicksum(W2_offset[i, j] for i in range(len(nn.W2)) for j in range(l2_size)) +
        gp.quicksum(W3_offset[i, j] for i in range(len(nn.W3)) for j in range(l3_size))
    )

    model.addConstr(objective <= 10000, "ObjectiveUpperBound")
    model.setObjective(objective, GRB.MAXIMIZE)
    model.setParam("Method", 2)

    model.setParam('TimeLimit', timeLimit)
    model.optimize()
    # model.setParam(GRB.Param.NumericFocus, 3)

    if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
        if model.SolCount == 0:
            print("Timeout")
            return
        W1_values = np.array([[nn.W1[i][j] for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_values = np.array([[nn.W2[i][j] for j in range(l2_size)] for i in range(len(nn.W2))])
        W3_values = np.array([[nn.W3[i][j] for j in range(l3_size)] for i in range(len(nn.W3))])
        b1_values = np.array([nn.b1[0, j] for j in range(l1_size)])
        b2_values = np.array([nn.b2[0, j] for j in range(l2_size)])
        b3_values = np.array([nn.b3[0, j] for j in range(l3_size)])
        
        W1_values_with_offset = np.array([[nn.W1[i][j] + W1_offset[i, j].X for j in range(l1_size)] for i in range(len(nn.W1))])
        W2_values_with_offset = np.array([[nn.W2[i][j] + W2_offset[i, j].X for j in range(l2_size)] for i in range(len(nn.W2))])
        W3_values_with_offset = np.array([[nn.W3[i][j] + W3_offset[i, j].X for j in range(l3_size)] for i in range(len(nn.W3))])
        b1_values_with_offset = np.array([nn.b1[0, j] + b1_offset[j].X for j in range(l1_size)])
        b2_values_with_offset = np.array([nn.b2[0, j] + b2_offset[j].X for j in range(l2_size)])
        b3_values_with_offset = np.array([nn.b3[0, j] + b3_offset[j].X for j in range(l3_size)])
        
        vw = VerifyWeights(n, l1, l2, [flp_idx], tol, W1_values, W2_values, W3_values, b1_values, b2_values, b3_values,
                    W1_values_with_offset, W2_values_with_offset, W3_values_with_offset,
                    b1_values_with_offset, b2_values_with_offset, b3_values_with_offset)
        vw.main()

        Z3_values = [[Z3[i, j].X for j in range(l3_size)] for i in range(len(X))]
        print(y.reshape(1,-1)[0])
        print("Z3:", Z3_values)


main()