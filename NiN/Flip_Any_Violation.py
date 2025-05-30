import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 7200

X_data = np.load("input_features_logits.npz")
X = X_data["X"]
Z3_target = X_data["Z3"]

W_data = np.load("nin_block_weights.npz")
fc1_w, fc1_b = W_data["fc1_w"], W_data["fc1_b"]
classifier_w, classifier_b = W_data["classifier_w"], W_data["classifier_b"]

n_samples = len(X)
l1_size = fc1_w.shape[0]
l2_size = classifier_w.shape[0]
input_size = fc1_w.shape[1]

model = gp.Model()
model.setParam("OutputFlag", 1)

# Only optimize fc1
fc1_w_offset = model.addVars(*fc1_w.shape, lb=-GRB.INFINITY, name="fc1_w_offset")
fc1_b_offset = model.addVars(l1_size, lb=-GRB.INFINITY, name="fc1_b_offset")

def add_relu(model, Z, A, h, name_prefix):
    for i in range(len(Z)):
        model.addConstr((h[i] == 1) >> (Z[i] >= 0), name=f"{name_prefix}_pos_{i}")
        model.addConstr((h[i] == 1) >> (A[i] == Z[i]), name=f"{name_prefix}_eq_{i}")
        model.addConstr((h[i] == 0) >> (Z[i] <= 0), name=f"{name_prefix}_neg_{i}")
        model.addConstr((h[i] == 0) >> (A[i] == 0), name=f"{name_prefix}_zero_{i}")

misclassified_flags = model.addVars(n_samples, vtype=GRB.BINARY, name="misclassified_flags")
epsilon = 3e-5

for s in range(n_samples):
    x_sample = X[s]
    label = np.argmax(Z3_target[s])

    # First FC layer
    Z1 = model.addVars(l1_size, lb=-GRB.INFINITY, name=f"Z1_{s}")
    A1 = model.addVars(l1_size, lb=0, name=f"A1_{s}")
    h1 = model.addVars(l1_size, vtype=GRB.BINARY, name=f"h1_{s}")
    for j in range(l1_size):
        expr = gp.LinExpr()
        for i in range(input_size):
            expr += (fc1_w[j, i] + fc1_w_offset[j, i]) * x_sample[i]
        expr += fc1_b[j] + fc1_b_offset[j]
        model.addConstr(Z1[j] == expr)
    add_relu(model, Z1, A1, h1, f"relu1_{s}")

    # Classifier layer (NOT optimized!)
    Z2 = model.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
    for j in range(l2_size):
        expr = gp.LinExpr()
        for i in range(l1_size):
            expr += classifier_w[j, i] * A1[i]
        expr += classifier_b[j]
        model.addConstr(Z2[j] == expr)

    violations = model.addVars(l2_size, vtype=GRB.BINARY, name=f"violations_{s}")
    for k in range(l2_size):
        if k != label:
            model.addConstr((violations[k] == 1) >> (Z2[label] <= Z2[k] - epsilon), name=f"violation_1flip_{s}_{k}")
            model.addConstr((violations[k] == 0) >> (Z2[label] >= Z2[k] + epsilon), name=f"violation_0flip_{s}_{k}")
        else:
            model.addConstr(violations[k] == 0, name=f"violation_0_{s}_{k}")

    model.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) >= misclassified_flags[s])
    model.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) <= (l2_size - 1) * misclassified_flags[s])

model.addConstr(gp.quicksum(misclassified_flags[s] for s in range(n_samples)) == 1, name="exactly_one_misclassified")

# Abs value constraints for offsets (L1 objective)
abs_fc1_w = model.addVars(*fc1_w.shape, lb=0, name="abs_fc1_w")
abs_fc1_b = model.addVars(l1_size, lb=0, name="abs_fc1_b")

for i in range(fc1_w.shape[0]):
    for j in range(fc1_w.shape[1]):
        model.addConstr(abs_fc1_w[i, j] >= fc1_w_offset[i, j])
        model.addConstr(abs_fc1_w[i, j] >= -fc1_w_offset[i, j])
for i in range(l1_size):
    model.addConstr(abs_fc1_b[i] >= fc1_b_offset[i])
    model.addConstr(abs_fc1_b[i] >= -fc1_b_offset[i])

objective = (
    gp.quicksum(abs_fc1_w[i, j] for i in range(fc1_w.shape[0]) for j in range(fc1_w.shape[1])) +
    gp.quicksum(abs_fc1_b[i] for i in range(l1_size))
)

model.setObjective(objective, GRB.MINIMIZE)
model.setParam("TimeLimit", timeLimit)
model.optimize()

if model.status == GRB.TIME_LIMIT or model.status == GRB.OPTIMAL:
    if model.SolCount == 0:
        print("Timeout: No feasible solution.")
    else:
        fc1_w_off = np.array([[fc1_w_offset[i, j].X for j in range(fc1_w.shape[1])] for i in range(fc1_w.shape[0])])
        fc1_b_off = np.array([fc1_b_offset[i].X for i in range(l1_size)])

        fc1_w_new = fc1_w + fc1_w_off
        fc1_b_new = fc1_b + fc1_b_off

        def relu(x): return np.maximum(0, x)

        misclassified_indices = []
        predictions = []
        labels = []
        for i in range(n_samples):
            label = np.argmax(Z3_target[i])
            x = X[i]

            z1 = fc1_w_new @ x + fc1_b_new
            a1 = relu(z1)
            z2 = classifier_w @ a1 + classifier_b

            pred = np.argmax(z2)
            predictions.append(pred)
            labels.append(label)

            correct = pred == label

            if not correct:
                misclassified_indices.append(i)
                print(f"Sample {i} misclassified: true={label}, pred={pred}")

        print("Predictions:", predictions)
        print("Labels:", labels)
        print(f"Misclassified: {len(misclassified_indices)}")

        print("FC1 weight change stats:")
        print("Max diff:", np.abs(fc1_w_new - fc1_w).max())
        print("Mean diff:", np.abs(fc1_w_new - fc1_w).mean())
        print("Number of changed elements:", np.sum(fc1_w_new != fc1_w))

else:
    print("No solution found.")
