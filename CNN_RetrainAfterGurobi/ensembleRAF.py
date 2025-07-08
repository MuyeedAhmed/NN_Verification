import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from CNNetworks import NIN_MNIST

timeLimit = 3600
n_samples_gurobi = 1000
dataset_name = "MNIST"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

def create_model_for_dataset(name):
    return NIN_MNIST(num_classes=10)

def GurobiBorder(dataset_name, X, labels, Z2_target, W1, b1, W2, b2, tol=5e-6):
    Z1 = np.maximum(0, X @ W1.T + b1)
    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    model_g = gp.Model()
    model_g.setParam('OutputFlag', 0)

    W2_offset = model_g.addVars(*W2.shape, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    b2_offset = model_g.addVars(l2_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    misclassified_flags = model_g.addVars(n_samples, vtype=GRB.BINARY)

    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        A1_fixed = Z1[s]
        Z2 = model_g.addVars(l2_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b2[j] + b2_offset[j]
            model_g.addConstr(Z2[j] == expr)

        violations = model_g.addVars(l2_size, vtype=GRB.BINARY)
        for k in range(l2_size):
            if k != label_max:
                model_g.addGenConstrIndicator(violations[k], True, Z2[label_max] <= Z2[k] - tol)
                model_g.addGenConstrIndicator(violations[k], False, Z2[label_max] >= Z2[k] + tol)
            else:
                model_g.addConstr(violations[k] == 0)

        model_g.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) >= misclassified_flags[s])
        model_g.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) <= (l2_size - 1) * misclassified_flags[s])

    # Ensure at least one flip
    model_g.addConstr(gp.quicksum(misclassified_flags[s] for s in range(n_samples)) >= 1)

    abs_W2 = model_g.addVars(*W2.shape, lb=0)
    abs_b2 = model_g.addVars(l2_size, lb=0)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            model_g.addConstr(abs_W2[i, j] >= W2_offset[i, j])
            model_g.addConstr(abs_W2[i, j] >= -W2_offset[i, j])
    for i in range(l2_size):
        model_g.addConstr(abs_b2[i] >= b2_offset[i])
        model_g.addConstr(abs_b2[i] >= -b2_offset[i])

    objective = (
        gp.quicksum(abs_W2[i, j] for i in range(W2.shape[0]) for j in range(W2.shape[1])) +
        gp.quicksum(abs_b2[i] for i in range(l2_size))
    )
    model_g.setObjective(objective, GRB.MINIMIZE)
    model_g.setParam('TimeLimit', timeLimit)
    model_g.optimize()

    if model_g.status == GRB.OPTIMAL and model_g.SolCount > 0:
        W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])
        print("||W2_offset||₁ =", np.sum(np.abs(W2_off)))
        print("||b2_offset||₁ =", np.sum(np.abs(b2_off)))
        return W2 + W2_off, b2 + b2_off
    else:
        return None

def apply_gurobi_patch(model, checkpoint_path, W2_new, b2_new, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    W2_tensor = torch.tensor(W2_new, dtype=torch.float32).to(device)
    b2_tensor = torch.tensor(b2_new, dtype=torch.float32).to(device)

    model.classifier.weight.data = W2_tensor
    model.classifier.bias.data = b2_tensor
    return model

def ensemble_predict(models, inputs):
    with torch.no_grad():
        logits_sum = None
        for model in models:
            model.eval()
            out = model(inputs)
            logits_sum = out if logits_sum is None else logits_sum + out
        avg_logits = logits_sum / len(models)
        return avg_logits.argmax(dim=1)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

X_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt").numpy()
labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt").numpy()
pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt").numpy()

W1 = torch.load(f"checkpoints/{dataset_name}/Run1_fc_hidden_weight.pt", map_location='cpu').numpy()
b1 = torch.load(f"checkpoints/{dataset_name}/Run1_fc_hidden_bias.pt", map_location='cpu').numpy()
W2 = torch.load(f"checkpoints/{dataset_name}/Run1_classifier_weight.pt", map_location='cpu').numpy()
b2 = torch.load(f"checkpoints/{dataset_name}/Run1_classifier_bias.pt", map_location='cpu').numpy()

X_parts = np.array_split(X_full[:n_samples_gurobi], 5)
labels_parts = np.array_split(labels_full[:n_samples_gurobi], 5)
Z2_target_full = np.maximum(0, X_full[:n_samples_gurobi] @ W1.T + b1) @ W2.T + b2
Z2_parts = np.array_split(Z2_target_full, 5)

ensemble_models = []
individual_accuracies = []

for i in range(5):
    result = GurobiBorder(dataset_name, X_parts[i], labels_parts[i], Z2_parts[i], W1, b1, W2, b2)
    if result is None:
        print(f"Gurobi failed on part {i}")
        continue
    W2_new, b2_new = result
    model_i = create_model_for_dataset(dataset_name).to(device)
    model_i = apply_gurobi_patch(model_i, f"./checkpoints/{dataset_name}/Run1_full_checkpoint.pth", W2_new, b2_new, device)
    ensemble_models.append(model_i)

    acc = evaluate_model(model_i, test_loader, device)
    individual_accuracies.append(acc)
    print(f"Model {i+1} Accuracy: {acc:.2f}%")

correct = 0
total = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    preds = ensemble_predict(ensemble_models, inputs)
    correct += preds.eq(labels).sum().item()
    total += labels.size(0)

ensemble_acc = 100. * correct / total
print(f"\nEnsemble Accuracy: {ensemble_acc:.2f}%")

print("\nSummary:")
for i, acc in enumerate(individual_accuracies):
    print(f"Model {i+1}: {acc:.2f}%")
print(f"Ensemble: {ensemble_acc:.2f}%")
