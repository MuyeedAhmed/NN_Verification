import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Load data
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

# Offsets
W1_offset = model.addVars(*W1.shape, lb=-GRB.INFINITY, name="W1_offset")
b1_offset = model.addVars(l1_size, lb=-GRB.INFINITY, name="b1_offset")
W2_offset = model.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
b2_offset = model.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")
W3_offset = model.addVars(*W3.shape, lb=-GRB.INFINITY, name="W3_offset")
b3_offset = model.addVars(l3_size, lb=-GRB.INFINITY, name="b3_offset")

# ReLU helper
def add_relu(model, Z, A, h, name_prefix):
    for i in range(len(Z)):
        model.addConstr((h[i] == 1) >> (Z[i] >= 0), name=f"{name_prefix}_pos_{i}")
        model.addConstr((h[i] == 1) >> (A[i] == Z[i]), name=f"{name_prefix}_eq_{i}")
        model.addConstr((h[i] == 0) >> (Z[i] <= 0), name=f"{name_prefix}_neg_{i}")
        model.addConstr((h[i] == 0) >> (A[i] == 0), name=f"{name_prefix}_zero_{i}")

# Binary: which sample is modified?
is_modified = model.addVars(n_samples, vtype=GRB.BINARY, name="is_modified")
model.addConstr(gp.quicksum(is_modified[s] for s in range(n_samples)) == 1)

epsilon = 1e-4
Z3_outputs = {}

for s in range(n_samples):
    x_sample = X[s]
    label = int(np.argmax(Z3_target[s]))
    second_label = int(np.argsort(Z3_target[s])[-2])

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
    Z3_outputs[s] = Z3
    for j in range(l3_size):
        expr = gp.LinExpr()
        for i in range(l2_size):
            expr += (W3[j, i] + W3_offset[j, i]) * A2[i]
        expr += b3[j] + b3_offset[j]
        model.addConstr(Z3[j] == expr)

    # for k in range(l3_size):
    #     if k != second_label:
    #         (is_modified[s] == 1) >> (Z3[second_label] >= Z3[k] + epsilon)
    #     if k != label:
    #         (is_modified[s] == 0) >> (Z3[label] >= Z3[k] + epsilon)

    for k in range(l3_size):
        if k != second_label:
            model.addConstr((is_modified[s] == 1) >> (Z3[second_label] >= Z3[k] + epsilon), name=f"Decision_{s}_{k}")
        if k != label:
            model.addConstr((is_modified[s] == 0) >> (Z3[label] >= Z3[k] + epsilon), name=f"Decision_{s}_{k}")

# Objective
abs_W1 = model.addVars(*W1.shape, lb=0)
abs_b1 = model.addVars(l1_size, lb=0)
abs_W2 = model.addVars(*W2.shape, lb=0)
abs_b2 = model.addVars(l2_size, lb=0)
abs_W3 = model.addVars(*W3.shape, lb=0)
abs_b3 = model.addVars(l3_size, lb=0)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        model.addConstr(abs_W1[i, j] >= W1_offset[i, j])
        model.addConstr(abs_W1[i, j] >= -W1_offset[i, j])
for i in range(l1_size):
    model.addConstr(abs_b1[i] >= b1_offset[i])
    model.addConstr(abs_b1[i] >= -b1_offset[i])
for i in range(W2.shape[0]):
    for j in range(W2.shape[1]):
        model.addConstr(abs_W2[i, j] >= W2_offset[i, j])
        model.addConstr(abs_W2[i, j] >= -W2_offset[i, j])
for i in range(l2_size):
    model.addConstr(abs_b2[i] >= b2_offset[i])
    model.addConstr(abs_b2[i] >= -b2_offset[i])
for i in range(W3.shape[0]):
    for j in range(W3.shape[1]):
        model.addConstr(abs_W3[i, j] >= W3_offset[i, j])
        model.addConstr(abs_W3[i, j] >= -W3_offset[i, j])
for i in range(l3_size):
    model.addConstr(abs_b3[i] >= b3_offset[i])
    model.addConstr(abs_b3[i] >= -b3_offset[i])

objective = (
    gp.quicksum(abs_W1[i, j] for i in range(W1.shape[0]) for j in range(W1.shape[1])) +
    gp.quicksum(abs_b1[i] for i in range(l1_size)) +
    gp.quicksum(abs_W2[i, j] for i in range(W2.shape[0]) for j in range(W2.shape[1])) +
    gp.quicksum(abs_b2[i] for i in range(l2_size)) +
    gp.quicksum(abs_W3[i, j] for i in range(W3.shape[0]) for j in range(W3.shape[1])) +
    gp.quicksum(abs_b3[i] for i in range(l3_size))
)

model.setObjective(objective, GRB.MINIMIZE)
model.setParam("TimeLimit", 10000)
model.optimize()

if model.status in [GRB.TIME_LIMIT, GRB.OPTIMAL]:
    if model.SolCount == 0:
        print("Timeout")
    else:
        print("\nModified sample (is_modified):")
        print([int(round(is_modified[s].X)) for s in range(n_samples)])

        print("\nZ3 values from Gurobi:")
        for s in range(n_samples):
            z3_vals = [Z3_outputs[s][j].X for j in range(l3_size)]
            print(f"Sample {s}: {np.round(z3_vals, 4)}")

        # Apply offsets
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


        # W1_new = W1
        # b1_new = b1
        # W2_new = W2
        # b2_new = b2
        # W3_new = W3
        # b3_new = b3

        def relu(x): return np.maximum(0, x)

        print("\nZ3 values from NumPy forward pass:")
        for i in range(n_samples):
            x = X[i]
            z1 = relu(W1_new @ x + b1_new)
            z2 = relu(W2_new @ z1 + b2_new)
            z3 = W3_new @ z2 + b3_new
            print(f"Sample {i}: {np.round(z3, 4)}")


        misclassified_indices = []
        predictions = []
        labels = []
        for i in range(n_samples):
            x = X[i]
            label = np.argmax(Z3_target[i])

            z1 = W1_new @ x + b1_new
            a1 = relu(z1)
            z2 = W2_new @ a1 + b2_new
            a2 = relu(z2)
            z3 = W3_new @ a2 + b3_new

            pred = np.argmax(z3)
            predictions.append(pred)
            labels.append(label)

            correct = pred == label

            if not correct:
                misclassified_indices.append(i)
                # print(f"Sample {i} misclassified: true={label}, pred={pred}")

        print("Predictions:", predictions)
        print("Labels:", labels)
        # print(f"Misclassified: {len(misclassified_indices)}")

else:
    print("No solution found.")
