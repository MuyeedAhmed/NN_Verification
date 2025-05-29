
import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 600
X_data = np.load("input_features_logits.npz")
X = X_data["X"]
Z3_target = X_data["Z3"]

W_data = np.load("cnn_fc_weights.npz")
W1, b1 = W_data["fc1_w"], W_data["fc1_b"]
W2, b2 = W_data["fc2_w"], W_data["fc2_b"]
W3, b3 = W_data["fc3_w"], W_data["fc3_b"]

n_samples = len(X)
l1_size = W1.shape[0]
l2_size = W2.shape[0]
l3_size = W3.shape[0]
input_size = W1.shape[1]

model = gp.Model()
model.setParam("OutputFlag", 1)

W1_offset = model.addVars(*W1.shape, lb=-GRB.INFINITY, name="W1_offset")
b1_offset = model.addVars(l1_size, lb=-GRB.INFINITY, name="b1_offset")
W2_offset = model.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
b2_offset = model.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")
W3_offset = model.addVars(*W3.shape, lb=-GRB.INFINITY, name="W3_offset")
b3_offset = model.addVars(l3_size, lb=-GRB.INFINITY, name="b3_offset")

def add_relu(model, Z, A, h, name_prefix):
    for i in range(len(Z)):
        model.addConstr((h[i] == 1) >> (Z[i] >= 0), name=f"{name_prefix}_pos_{i}")
        model.addConstr((h[i] == 1) >> (A[i] == Z[i]), name=f"{name_prefix}_eq_{i}")
        model.addConstr((h[i] == 0) >> (Z[i] <= 0), name=f"{name_prefix}_neg_{i}")
        model.addConstr((h[i] == 0) >> (A[i] == 0), name=f"{name_prefix}_zero_{i}")

Z3_list = []
max_min_diff = []

for s in range(n_samples):
    label_max = int(np.argmax(Z3_target[s]))
    label_min = int(np.argmin(Z3_target[s]))
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
    A2 = model.addVars(l2_size, lb=0, name=f"A2_{s}")
    h2 = model.addVars(l2_size, vtype=GRB.BINARY, name=f"h2_{s}")
    for j in range(l2_size):
        expr = gp.LinExpr()
        for i in range(l1_size):
            expr += (W2[j, i] + W2_offset[j, i]) * A1[i]
        expr += b2[j] + b2_offset[j]
        model.addConstr(Z2[j] == expr)
    add_relu(model, Z2, A2, h2, f"relu2_{s}")

    Z3 = model.addVars(l3_size, lb=-GRB.INFINITY, name=f"Z3_{s}")
    for j in range(l3_size):
        expr = gp.LinExpr()
        for i in range(l2_size):
            expr += (W3[j, i] + W3_offset[j, i]) * A2[i]
        expr += b3[j] + b3_offset[j]
        model.addConstr(Z3[j] == expr)

    for k in range(l3_size):
        if k != label_max:
            model.addConstr(Z3[label_max] >= Z3[k] + 3e-5, f"Z3_max_{s}_{k}")

    Z3_list.append(Z3)
    max_min_diff.append(Z3[label_max]-Z3[label_min])

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
        W3_off = np.array([[W3_offset[i, j].X for j in range(W3.shape[1])] for i in range(W3.shape[0])])
        b3_off = np.array([b3_offset[i].X for i in range(l3_size)])
        W1_new = W1 + W1_off
        b1_new = b1 + b1_off
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off
        W3_new = W3 + W3_off
        b3_new = b3 + b3_off

        # print(Z3_target)
        # for s in range(n_samples):
        #     z3_vals = [Z3_list[s][j].X for j in range(l3_size)]
        #     print(f"Z3[{s}]:", z3_vals)
        print(np.sum(np.abs(W1_off)), np.sum(np.abs(b1_off)),
              np.sum(np.abs(W2_off)), np.sum(np.abs(b2_off)),
              np.sum(np.abs(W3_off)), np.sum(np.abs(b3_off)))
        
        print("Weights and biases adjusted successfully.")
        print("Objective value:", model.ObjVal)
        def relu(x): return np.maximum(0, x)
        misclassified_indices = []
        predictions = []
        labels = []
        for i in range(n_samples):
            x = X[i]
            label = int(np.argmax(Z3_target[i]))
            z1 = W1_new @ x + b1_new
            a1 = relu(z1)
            z2 = W2_new @ a1 + b2_new
            a2 = relu(z2)
            z3 = W3_new @ a2 + b3_new
            print(z3)
            pred = np.argmax(z3)
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
            label = int(np.argmax(Z3_target[i]))

            # --- Forward pass to get z3 ---
            z1 = W1_new @ x + b1_new
            a1 = relu(z1)
            z2 = W2_new @ a1 + b2_new
            a2 = relu(z2)
            z3 = W3_new @ a2 + b3_new

            # --- Compute softmax for both Z3_target and z3 ---
            def softmax(logits):
                e = np.exp(logits - np.max(logits))  # stability
                return e / np.sum(e)

            pred_probs = softmax(z3)
            target_probs = softmax(Z3_target[i])

            # --- Cross-entropy loss against ground-truth label ---
            ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
            ce_loss_target += -np.log(target_probs[label] + 1e-12)

        print("Average CE loss (Z3 vs labels):", ce_loss_target / n_samples)
        print("Average CE loss (z3 vs labels):", ce_loss_pred / n_samples)

else:
    print("No solution found.")
