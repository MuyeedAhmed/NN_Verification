import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 7200

X_data = np.load("input_features_logits.npz")
X = X_data["X"]
B2_target = X_data["B2"]

W_data = np.load("nin_block_weights.npz")
block1_conv0_w, block1_conv0_b = W_data["block1_conv0_w"], W_data["block1_conv0_b"]
block1_conv2_w, block1_conv2_b = W_data["block1_conv2_w"], W_data["block1_conv2_b"]
block1_conv4_w, block1_conv4_b = W_data["block1_conv4_w"], W_data["block1_conv4_b"]
block2_conv0_w, block2_conv0_b = W_data["block2_conv0_w"], W_data["block2_conv0_b"]
block2_conv2_w, block2_conv2_b = W_data["block2_conv2_w"], W_data["block2_conv2_b"]
block2_conv4_w, block2_conv4_b = W_data["block2_conv4_w"], W_data["block2_conv4_b"]

n_samples = len(X)

block1_conv0_size = block1_conv0_w.shape[0]
block1_conv2_size = block1_conv2_w.shape[0]
block1_conv4_size = block1_conv4_w.shape[0]
block2_conv0_size = block2_conv0_w.shape[0]
block2_conv2_size = block2_conv2_w.shape[0]
block2_conv4_size = block2_conv4_w.shape[0]

input_size = block1_conv0_w.shape[1]

model = gp.Model()
model.setParam("OutputFlag", 1)

block1_conv0_offset = model.addVars(*block1_conv0_size.shape, lb=-GRB.INFINITY, name="block1_conv0_offset")
block1_conv0_b_offset = model.addVars(block1_conv0_size, lb=-GRB.INFINITY, name="b1_conv0_offset")
block1_conv2_offset = model.addVars(*block1_conv2_size.shape, lb=-GRB.INFINITY, name="block1_conv2_offset")
block1_conv2_b_offset = model.addVars(block1_conv2_size, lb=-GRB.INFINITY, name="b1_conv2_offset")
block1_conv4_offset = model.addVars(*block1_conv4_size.shape, lb=-GRB.INFINITY, name="block1_conv4_offset")
block1_conv4_b_offset = model.addVars(block1_conv4_size, lb=-GRB.INFINITY, name="b1_conv4_offset")
block2_conv0_offset = model.addVars(*block2_conv0_size.shape, lb=-GRB.INFINITY, name="block2_conv0_offset")
block2_conv0_b_offset = model.addVars(block2_conv0_size, lb=-GRB.INFINITY, name="b2_conv0_offset")
block2_conv2_offset = model.addVars(*block2_conv2_size.shape, lb=-GRB.INFINITY, name="block2_conv2_offset")
block2_conv2_b_offset = model.addVars(block2_conv2_size, lb=-GRB.INFINITY, name="b2_conv2_offset")
block2_conv4_offset = model.addVars(*block2_conv4_size.shape, lb=-GRB.INFINITY, name="block2_conv4_offset")
block2_conv4_b_offset = model.addVars(block2_conv4_size, lb=-GRB.INFINITY, name="b2_conv4_offset")


def add_relu(model, Z, A, h, name_prefix):
    for i in range(len(Z)):
        model.addConstr((h[i] == 1) >> (Z[i] >= 0), name=f"{name_prefix}_pos_{i}")
        model.addConstr((h[i] == 1) >> (A[i] == Z[i]), name=f"{name_prefix}_eq_{i}")
        model.addConstr((h[i] == 0) >> (Z[i] <= 0), name=f"{name_prefix}_neg_{i}")
        model.addConstr((h[i] == 0) >> (A[i] == 0), name=f"{name_prefix}_zero_{i}")

misclassified_flags = model.addVars(n_samples, vtype=GRB.BINARY, name="misclassified_flags")
epsilon = 1e-6

for s in range(n_samples):
    x_sample = X[s]
    label = np.argmax(B2_target[s])
    label_2nd = np.argsort(B2_target[s])[-2]

    Z1 = model.addVars(block1_conv0_size, lb=-GRB.INFINITY, name=f"Z1_{s}")
    A1 = model.addVars(block1_conv0_size, lb=0, name=f"A1_{s}")
    h1 = model.addVars(block1_conv0_size, vtype=GRB.BINARY, name=f"h1_{s}")
    for j in range(block1_conv0_size):
        expr = gp.LinExpr()
        for i in range(input_size):
            expr += (block1_conv0_w[j, i] + block1_conv0_offset[j, i]) * x_sample[i]
        expr += block1_conv0_b[j] + block1_conv0_b_offset[j]
        model.addConstr(Z1[j] == expr)
    add_relu(model, Z1, A1, h1, f"relu1_{s}")

    Z2 = model.addVars(block1_conv2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
    A2 = model.addVars(block1_conv2_size, lb=0, name=f"A2_{s}")
    h2 = model.addVars(block1_conv2_size, vtype=GRB.BINARY, name=f"h2_{s}")
    for j in range(block1_conv2_size):
        expr = gp.LinExpr()
        for i in range(block1_conv0_size):
            expr += (block1_conv2_w[j, i] + block1_conv2_offset[j, i]) * A1[i]
        expr += block1_conv2_b[j] + block1_conv2_b_offset[j]
        model.addConstr(Z2[j] == expr)
    add_relu(model, Z2, A2, h2, f"relu2_{s}")

    Z3 = model.addVars(block1_conv4_size, lb=-GRB.INFINITY, name=f"Z3_{s}")
    for j in range(block1_conv4_size):
        expr = gp.LinExpr()
        for i in range(block1_conv2_size):
            expr += (block1_conv2_w[j, i] + block1_conv2_offset[j, i]) * A2[i]
        expr += block1_conv2_b[j] + block1_conv2_b_offset[j]
        model.addConstr(Z3[j] == expr)


    violations = model.addVars(l3_size, vtype=GRB.BINARY, name=f"violations_{s}")
    for k in range(l3_size):
        if k != label:
            model.addConstr((violations[k] == 1) >> (Z3[label] <= Z3[k] - epsilon), name=f"violation_1flip_{s}_{k}")
            model.addConstr((violations[k] == 0) >> (Z3[label] >= Z3[k] + epsilon), name=f"violation_0flip_{s}_{k}")
        else:
            model.addConstr(violations[k] == 0, name=f"violation_0_{s}_{k}")

    model.addConstr(gp.quicksum(violations[k] for k in range(l3_size)) >= misclassified_flags[s])
    model.addConstr(gp.quicksum(violations[k] for k in range(l3_size)) <= (l3_size - 1) * misclassified_flags[s])


model.addConstr(gp.quicksum(misclassified_flags[s] for s in range(n_samples)) == 1, name="exactly_one_misclassified")

abs_W1 = model.addVars(*W1.shape, lb=0, name="abs_W1")
abs_b1 = model.addVars(l1_size, lb=0, name="abs_b1")
abs_W2 = model.addVars(*W2.shape, lb=0, name="abs_W2")
abs_b2 = model.addVars(l2_size, lb=0, name="abs_b2")
abs_W3 = model.addVars(*W3.shape, lb=0, name="abs_W3")
abs_b3 = model.addVars(l3_size, lb=0, name="abs_b3")

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
model.setParam("TimeLimit", timeLimit)
model.optimize()

if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
    if model.SolCount == 0:
        print("Timeout: No feasible solution.")
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

        def relu(x): return np.maximum(0, x)

        misclassified_indices = []
        predictions = []
        labels = []
        for i in range(n_samples):
            label = np.argmax(Z3_target[i])
            x = X[i]

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
                print(f"Sample {i} misclassified: true={label}, pred={pred}")

        print("Predictions:", predictions)
        print("Labels:", labels)
        print(f"Misclassified: {len(misclassified_indices)}")

else:
    print("No solution found.")
