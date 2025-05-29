import gurobipy as gp
from gurobipy import GRB
import numpy as np

timeLimit = 7200

X_data = np.load("input_features_logits.npz")
X = X_data["X"]
Z3_target = X_data["Z3"]

W_data = np.load("nin_block_weights.npz")
fc1_w, fc1_b = W_data["fc1_w"], W_data["fc1_b"]
fc2_w, fc2_b = W_data["fc2_w"], W_data["fc2_b"]
classifier_w, classifier_b = W_data["classifier_w"], W_data["classifier_b"]

n_samples = len(X)
l1_size = fc1_w.shape[0]
l2_size = fc2_w.shape[0]
l3_size = classifier_w.shape[0]
input_size = fc1_w.shape[1]

model = gp.Model()
model.setParam("OutputFlag", 1)

fc1_w_offset = model.addVars(*fc1_w.shape, lb=-GRB.INFINITY, name="fc1_w_offset")
fc1_b_offset = model.addVars(l1_size, lb=-GRB.INFINITY, name="fc1_b_offset")
fc2_w_offset = model.addVars(*fc2_w.shape, lb=-GRB.INFINITY, name="fc2_w_offset")
fc2_b_offset = model.addVars(l2_size, lb=-GRB.INFINITY, name="fc2_b_offset")

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
    label = np.argmax(Z3_target[s])

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

    Z2 = model.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
    A2 = model.addVars(l2_size, lb=0, name=f"A2_{s}")
    h2 = model.addVars(l2_size, vtype=GRB.BINARY, name=f"h2_{s}")
    for j in range(l2_size):
        expr = gp.LinExpr()
        for i in range(l1_size):
            expr += (fc2_w[j, i] + fc2_w_offset[j, i]) * A1[i]
        expr += fc2_b[j] + fc2_b_offset[j]
        model.addConstr(Z2[j] == expr)
    add_relu(model, Z2, A2, h2, f"relu2_{s}")

    Z3 = model.addVars(l3_size, lb=-GRB.INFINITY, name=f"Z3_{s}")
    for j in range(l3_size):
        expr = gp.LinExpr()
        for i in range(l2_size):
            expr += classifier_w[j, i] * A2[i]
        expr += classifier_b[j]
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

abs_fc1_w = model.addVars(*fc1_w.shape, lb=0, name="abs_fc1_w")
abs_fc1_b = model.addVars(l1_size, lb=0, name="abs_fc1_b")
abs_fc2_w = model.addVars(*fc2_w.shape, lb=0, name="abs_fc2_w")
abs_fc2_b = model.addVars(l2_size, lb=0, name="abs_fc2_b")

for i in range(fc1_w.shape[0]):
    for j in range(fc1_w.shape[1]):
        model.addConstr(abs_fc1_w[i, j] >= fc1_w_offset[i, j])
        model.addConstr(abs_fc1_w[i, j] >= -fc1_w_offset[i, j])
for i in range(l1_size):
    model.addConstr(abs_fc1_b[i] >= fc1_b_offset[i])
    model.addConstr(abs_fc1_b[i] >= -fc1_b_offset[i])
for i in range(fc2_w.shape[0]):
    for j in range(fc2_w.shape[1]):
        model.addConstr(abs_fc2_w[i, j] >= fc2_w_offset[i, j])
        model.addConstr(abs_fc2_w[i, j] >= -fc2_w_offset[i, j])
for i in range(l2_size):
    model.addConstr(abs_fc2_b[i] >= fc2_b_offset[i])
    model.addConstr(abs_fc2_b[i] >= -fc2_b_offset[i])

objective = (
    gp.quicksum(abs_fc1_w[i, j] for i in range(fc1_w.shape[0]) for j in range(fc1_w.shape[1])) +
    gp.quicksum(abs_fc1_b[i] for i in range(l1_size)) +
    gp.quicksum(abs_fc2_w[i, j] for i in range(fc2_w.shape[0]) for j in range(fc2_w.shape[1])) +
    gp.quicksum(abs_fc2_b[i] for i in range(l2_size))
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
        fc2_w_off = np.array([[fc2_w_offset[i, j].X for j in range(fc2_w.shape[1])] for i in range(fc2_w.shape[0])])
        fc2_b_off = np.array([fc2_b_offset[i].X for i in range(l2_size)])

        fc1_w_new = fc1_w + fc1_w_off
        fc1_b_new = fc1_b + fc1_b_off
        fc2_w_new = fc2_w + fc2_w_off
        fc2_b_new = fc2_b + fc2_b_off

        def relu(x): return np.maximum(0, x)

        misclassified_indices = []
        predictions = []
        labels = []
        for i in range(n_samples):
            label = np.argmax(Z3_target[i])
            x = X[i]

            z1 = fc1_w_new @ x + fc1_b_new
            a1 = relu(z1)
            z2 = fc2_w_new @ a1 + fc2_b_new
            a2 = relu(z2)
            z3 = classifier_w @ a2 + classifier_b

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


print("FC1 weight change stats:")
print("  Max diff:", np.abs(fc1_w_new - fc1_w).max())
print("  Mean diff:", np.abs(fc1_w_new - fc1_w).mean())
print("  Number of changed elements:", np.sum(fc1_w_new != fc1_w))

print("FC1 bias change stats:")
print("  Max diff:", np.abs(fc1_b_new - fc1_b).max())
print("  Mean diff:", np.abs(fc1_b_new - fc1_b).mean())
print("  Number of changed elements:", np.sum(fc1_b_new != fc1_b))

print("FC2 weight change stats:")
print("  Max diff:", np.abs(fc2_w_new - fc2_w).max())
print("  Mean diff:", np.abs(fc2_w_new - fc2_w).mean())
print("  Number of changed elements:", np.sum(fc2_w_new != fc2_w))

print("FC2 bias change stats:")
print("  Max diff:", np.abs(fc2_b_new - fc2_b).max())
print("  Mean diff:", np.abs(fc2_b_new - fc2_b).mean())
print("  Number of changed elements:", np.sum(fc2_b_new != fc2_b))