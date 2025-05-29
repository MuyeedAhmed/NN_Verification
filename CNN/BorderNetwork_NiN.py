
import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 18000

import torch
import numpy as np

X_fc = torch.load("fc_data/X_fc.pt")[0:40].numpy()
labels = torch.load("fc_data/labels.pt")[0:40].numpy()
print("labels", labels)

weights = torch.load("fc_data/weights.pt")
W1 = weights["fc_hidden_weight"].numpy()
b1 = weights["fc_hidden_bias"].numpy()
W2 = weights["classifier_weight"].numpy()
b2 = weights["classifier_bias"].numpy()


print(X_fc)
Z1 = np.maximum(0, X_fc @ W1.T + b1)
Z2 = Z1 @ W2.T + b2
# print(Z2)
np.savez("input_features_logits.npz", X=X_fc, Z2=Z2)
np.savez("cnn_fc_weights.npz", fc1_w=W1, fc1_b=b1, fc2_w=W2, fc2_b=b2)



X_data = np.load("input_features_logits.npz")
X = X_data["X"]
Z2_target = X_data["Z2"]
for i in range(len(Z2_target)):
    print("Z2_target", np.argmax(Z2_target[i]))


W_data = np.load("cnn_fc_weights.npz")
W1, b1 = W_data["fc1_w"], W_data["fc1_b"]
W2, b2 = W_data["fc2_w"], W_data["fc2_b"]

n_samples = len(X)
l1_size = W1.shape[0]
l2_size = W2.shape[0]
input_size = W1.shape[1]

model = gp.Model()
model.setParam("OutputFlag", 1)

W1_offset = model.addVars(*W1.shape, lb=-GRB.INFINITY, name="W1_offset")
b1_offset = model.addVars(l1_size, lb=-GRB.INFINITY, name="b1_offset")
W2_offset = model.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
b2_offset = model.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")


def relu(x): return np.maximum(0, x)

def add_relu(model, Z, A, h, name_prefix):
    for i in range(len(Z)):
        model.addConstr((h[i] == 1) >> (Z[i] >= 0), name=f"{name_prefix}_pos_{i}")
        model.addConstr((h[i] == 1) >> (A[i] == Z[i]), name=f"{name_prefix}_eq_{i}")
        model.addConstr((h[i] == 0) >> (Z[i] <= 0), name=f"{name_prefix}_neg_{i}")
        model.addConstr((h[i] == 0) >> (A[i] == 0), name=f"{name_prefix}_zero_{i}")

Z2_list = []
max_min_diff = []

for s in range(n_samples):
    label_max = int(np.argmax(Z2_target[s]))
    label_min = int(np.argmin(Z2_target[s]))
    x_sample = X[s]
    Z1 = model.addVars(l1_size, lb=-GRB.INFINITY, name=f"Z1_{s}")
    A1 = model.addVars(l1_size, lb=0, name=f"A1_{s}")
    h1 = model.addVars(l1_size, vtype=GRB.BINARY, name=f"h1_{s}")
    for j in range(l1_size):
        expr = gp.LinExpr()
        for i in range(input_size):
            expr += (W1[j, i] + W1_offset[j, i]) * x_sample[i]
        expr += b1[j] + b1_offset[j]
        model.addConstr(Z1[j] == expr)
    add_relu(model, Z1, A1, h1, f"relu1_{s}")
    
    Z2 = model.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
    for j in range(l2_size):
        expr = gp.LinExpr()
        for i in range(l1_size):
            expr += (W2[j, i] + W2_offset[j, i]) * A1[i]
        expr += b2[j] + b2_offset[j]
        model.addConstr(Z2[j] == expr)

    for k in range(l2_size):
        if k != label_max:
            model.addConstr(Z2[label_max] >= Z2[k] + 3e-5, f"Z2_max_{s}_{k}")

    Z2_list.append(Z2)
    max_min_diff.append(Z2[label_max]-Z2[label_min])

objective = gp.quicksum(max_min_diff)
model.setObjective(objective, GRB.MINIMIZE)

model.addConstr(gp.quicksum(max_min_diff) >= 0, "ObjectiveLowerBound")
model.setParam('TimeLimit', timeLimit)
model.optimize()

if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
    if model.SolCount == 0:
        print("Timeout")
    else:
        W1_off = np.array([[W1_offset[i, j].X for j in range(W1.shape[1])] for i in range(W1.shape[0])])
        b1_off = np.array([b1_offset[i].X for i in range(l1_size)])
        W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])
        W1_new = W1 + W1_off
        b1_new = b1 + b1_off
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off

        # print(Z2_target)
        # for s in range(n_samples):
        #     z2_vals = [Z2_list[s][j].X for j in range(l3_size)]
        #     print(f"Z2[{s}]:", z2_vals)
        print(np.sum(np.abs(W1_off)), np.sum(np.abs(b1_off)),
              np.sum(np.abs(W2_off)), np.sum(np.abs(b2_off)))
        
        print("Weights and biases adjusted successfully.")
        print("Objective value:", model.ObjVal)
        def relu(x): return np.maximum(0, x)
        misclassified_indices = []
        predictions = []
        labels = []
        for i in range(n_samples):
            x = X[i]
            label = int(np.argmax(Z2_target[i]))
            z1 = W1_new @ x + b1_new
            a1 = relu(z1)
            z2 = W2_new @ a1 + b2_new
            # print(z2)
            pred = np.argmax(z2)
            predictions.append(pred)
            labels.append(label)
            if pred != label:
                misclassified_indices.append(i)
                print(f"Sample {i} misclassified: true={label}, pred={pred}")
        print("Predictions:", predictions)
        print("Labels:", labels)
        print(f"Misclassified: {len(misclassified_indices)}")
        
        ce_loss_target = 0
        ce_loss_pred = 0

        for i in range(n_samples):
            x = X[i]
            label = int(np.argmax(Z2_target[i]))
            z1 = W1_new @ x + b1_new
            a1 = relu(z1)
            z2 = W2_new @ a1 + b2_new
            
            def softmax(logits):
                e = np.exp(logits - np.max(logits))
                return e / np.sum(e)

            pred_probs = softmax(z2)
            target_probs = softmax(Z2_target[i])

            ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
            ce_loss_target += -np.log(target_probs[label] + 1e-12)

        print("Average Cross Entropy loss (Z2 vs labels):", ce_loss_target / n_samples)
        print("Average Cross Entropy loss (z2 vs labels):", ce_loss_pred / n_samples)

else:
    print("No solution found.")
