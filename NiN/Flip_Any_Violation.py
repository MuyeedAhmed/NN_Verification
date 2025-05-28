import gurobipy as gp
from gurobipy import GRB
import numpy as np


X_data = np.load("input_features_logits.npz")  
X = X_data["X"]
Z3_target = X_data["Z3"]


W_data = np.load("nin_block_weights.npz")
block1_conv0_w, block1_conv0_b = W_data["block1_conv0_w"], W_data["block1_conv0_b"]
block1_conv2_w, block1_conv2_b = W_data["block1_conv2_w"], W_data["block1_conv2_b"]
block1_conv4_w, block1_conv4_b = W_data["block1_conv4_w"], W_data["block1_conv4_b"]
block2_conv0_w, block2_conv0_b = W_data["block2_conv0_w"], W_data["block2_conv0_b"]
block2_conv2_w, block2_conv2_b = W_data["block2_conv2_w"], W_data["block2_conv2_b"]
block2_conv4_w, block2_conv4_b = W_data["block2_conv4_w"], W_data["block2_conv4_b"]
classifier_conv_w = W_data["classifier_conv_w"]
classifier_conv_b = W_data["classifier_conv_b"]

n_samples = len(X)
out_ch, in_ch, kh, kw = block2_conv4_w.shape

model = gp.Model()
model.setParam("OutputFlag", 1)

block2_conv4_w_offset = model.addVars(*block2_conv4_w.shape, lb=-GRB.INFINITY, name="block2_conv4_w_offset")
block2_conv4_b_offset = model.addVars(block2_conv4_b.shape[0], lb=-GRB.INFINITY, name="block2_conv4_b_offset")

epsilon = 1e-6
misclassified_flags = model.addVars(n_samples, vtype=GRB.BINARY, name="misclassified_flags")


for s in range(n_samples):
    
    x_sample = X[s]
    label = np.argmax(Z3_target[s])
    
    label_2nd = np.argsort(Z3_target[s])[-2]
    

    
    block2_out = {}
    for out_c in range(16):
        for i in range(4):
            for j in range(4):
                expr = gp.LinExpr()
                for in_c in range(16):
                    #convolution
                    expr += (block2_conv4_w[out_c, in_c, 0, 0] + block2_conv4_w_offset[out_c, in_c, 0, 0]) * x_sample[in_c, i, j]
                expr += block2_conv4_b[out_c] + block2_conv4_b_offset[out_c]
                # ReLU
                var = model.addVar(lb=0, name=f"block2relu_{s}_{out_c}_{i}_{j}")
                model.addConstr(var >= expr)
                model.addConstr(var >= 0)
                block2_out[(out_c, i, j)] = var

   
    Z3 = {}
    for cls in range(classifier_conv_w.shape[0]):
        expr = gp.LinExpr()
        for c in range(16):
            for i in range(4):
                for j in range(4):
                    expr += classifier_conv_w[cls, c, 0, 0] * block2_out[(c, i, j)]
        expr += classifier_conv_b[cls]
        Z3[cls] = expr

    violations = model.addVars(classifier_conv_w.shape[0], vtype=GRB.BINARY, name=f"violations_{s}")
    for k in range(classifier_conv_w.shape[0]):
        if k != label:
            model.addConstr((violations[k] == 1) >> (Z3[label] <= Z3[k] - epsilon), name=f"violation_1flip_{s}_{k}")
            model.addConstr((violations[k] == 0) >> (Z3[label] >= Z3[k] + epsilon), name=f"violation_0flip_{s}_{k}")
        else:
            model.addConstr(violations[k] == 0, name=f"violation_0_{s}_{k}")

    model.addConstr(gp.quicksum(violations[k] for k in range(classifier_conv_w.shape[0])) >= misclassified_flags[s])
    model.addConstr(gp.quicksum(violations[k] for k in range(classifier_conv_w.shape[0])) <= (classifier_conv_w.shape[0] - 1) * misclassified_flags[s])


model.addConstr(gp.quicksum(misclassified_flags[s] for s in range(n_samples)) == 1, name="exactly_one_misclassified")


abs_w = model.addVars(*block2_conv4_w.shape, lb=0, name="abs_block2_conv4_w")
abs_b = model.addVars(block2_conv4_b.shape[0], lb=0, name="abs_block2_conv4_b")
for i in range(block2_conv4_w.shape[0]):
    for j in range(block2_conv4_w.shape[1]):
        model.addConstr(abs_w[i, j, 0, 0] >= block2_conv4_w_offset[i, j, 0, 0])
        model.addConstr(abs_w[i, j, 0, 0] >= -block2_conv4_w_offset[i, j, 0, 0])
for i in range(block2_conv4_b.shape[0]):
    model.addConstr(abs_b[i] >= block2_conv4_b_offset[i])
    model.addConstr(abs_b[i] >= -block2_conv4_b_offset[i])

objective = gp.quicksum(abs_w[i, j, 0, 0] for i in range(block2_conv4_w.shape[0]) for j in range(block2_conv4_w.shape[1])) + \
            gp.quicksum(abs_b[i] for i in range(block2_conv4_b.shape[0]))
model.setObjective(objective, GRB.MINIMIZE)

model.setParam("TimeLimit", 7200)
model.optimize()

optimized_w_offset = np.zeros_like(block2_conv4_w)
optimized_b_offset = np.zeros_like(block2_conv4_b)

for i in range(block2_conv4_w.shape[0]):
    for j in range(block2_conv4_w.shape[1]):
        optimized_w_offset[i, j, 0, 0] = block2_conv4_w_offset[i, j, 0, 0].X
for i in range(block2_conv4_b.shape[0]):
    optimized_b_offset[i] = block2_conv4_b_offset[i].X


optimized_w = block2_conv4_w + optimized_w_offset
optimized_b = block2_conv4_b + optimized_b_offset


print("Max weight diff:", np.abs(optimized_w - block2_conv4_w).max())
print("Max bias diff:", np.abs(optimized_b - block2_conv4_b).max())


print("Weight changes:\n", optimized_w - block2_conv4_w)
print("Bias changes:\n", optimized_b - block2_conv4_b)