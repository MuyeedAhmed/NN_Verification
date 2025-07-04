import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch

timeLimit = 18000

X = torch.load("CIFER_1E/X_fc.pt")[0:2000].numpy()
labels = torch.load("CIFER_1E/labels.pt")[0:2000].numpy()
# X = torch.load("CIFER_1E/X_fc.pt").numpy()
# labels = torch.load("CIFER_1E/labels.pt").numpy()
weights = torch.load("CIFER_1E/weights.pt")

W1 = weights["fc_hidden_weight"].numpy()
b1 = weights["fc_hidden_bias"].numpy()
W2 = weights["classifier_weight"].numpy()
b2 = weights["classifier_bias"].numpy()

Z1_fixed = np.maximum(0, X @ W1.T + b1)
Z2_target = Z1_fixed @ W2.T + b2

W_data = np.load("cnn_fc_weights.npz")
W2, b2 = W_data["fc2_w"], W_data["fc2_b"]
Z1_fixed = np.maximum(0, X @ W_data["fc1_w"].T + W_data["fc1_b"])

n_samples = len(X)
l1_size = W2.shape[1]
l2_size = W2.shape[0]

model = gp.Model()
model.setParam("OutputFlag", 1)

W2_offset = model.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
b2_offset = model.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")

Z2_list = []
max_min_diff = []

for s in range(n_samples):
    label_max = int(np.argmax(Z2_target[s]))
    label_min = int(np.argmin(Z2_target[s]))
    A1_fixed = Z1_fixed[s]

    Z2 = model.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
    for j in range(l2_size):
        expr = gp.LinExpr()
        for i in range(l1_size):
            expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
        expr += b2[j] + b2_offset[j]
        model.addConstr(Z2[j] == expr)

    for k in range(l2_size):
        if k != label_max:
            model.addConstr(Z2[label_max] >= Z2[k] + 3e-5, f"Z2_max_{s}_{k}")

    Z2_list.append(Z2)
    max_min_diff.append(Z2[label_max] - Z2[label_min])

objective = gp.quicksum(max_min_diff)
model.setObjective(objective, GRB.MINIMIZE)
model.addConstr(objective >= 0, "ObjectiveLowerBound")
model.setParam('TimeLimit', timeLimit)
model.optimize()

if model.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and model.SolCount > 0:
    W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
    b2_off = np.array([b2_offset[i].X for i in range(l2_size)])
    W2_new = W2 + W2_off
    b2_new = b2 + b2_off

    print("-------Weight/Bias Offsets-------")
    print("W2 offsets:", np.sum(np.abs(W2_off)))
    print("b2 offsets:", np.sum(np.abs(b2_off)))
    print("Objective value:", model.ObjVal)
    print("------------------------------------")
    def relu(x): return np.maximum(0, x)
    def softmax(logits):
        e = np.exp(logits - np.max(logits))
        return e / np.sum(e)

    misclassified = 0
    ce_loss_target = 0
    ce_loss_pred = 0
    predictions, true_labels = [], []

    for i in range(n_samples):
        x = X[i]
        label = int(np.argmax(Z2_target[i]))
        a1 = relu(W1 @ x + b1)
        z2 = W2_new @ a1 + b2_new
        pred = np.argmax(z2)

        predictions.append(pred)
        true_labels.append(label)
        if pred != label:
            print(f"Sample {i} misclassified: true={label}, pred={pred}")
            misclassified += 1

        pred_probs = softmax(z2)
        target_probs = softmax(Z2_target[i])
        ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
        ce_loss_target += -np.log(target_probs[label] + 1e-12)

    # print("Predictions:", predictions)
    # print("Labels:", true_labels)
    print(f"Misclassified: {misclassified}")
    print("Average Cross Entropy loss (Z2 vs labels):", ce_loss_target / n_samples)
    print("Average Cross Entropy loss (z2 vs labels):", ce_loss_pred / n_samples)

else:
    print("No solution found.")
