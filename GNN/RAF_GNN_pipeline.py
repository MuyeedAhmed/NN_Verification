import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gurobipy as gp
from gurobipy import GRB


timeLimit = 3600

class SimpleGNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc_hidden = nn.Linear(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, extract_fc_input=False):
        x = F.relu(self.fc_hidden(x))
        if extract_fc_input:
            return x, self.classifier(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.classifier(x), dim=1)


class RAF:
    def __init__(self, dataset_name, model, train_loader, test_loader, device, num_epochs=200, resume_epochs=100, batch_size=64, learning_rate=0.01, phase="Train"):
        self.dataset_name = dataset_name
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.resume_epochs = resume_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.phase = phase
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.NLLLoss()
        self.log_file = f"Stats_RAF/{self.dataset_name}_log.csv"
        os.makedirs("Stats_RAF", exist_ok=True)

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
                total_samples += labels.size(0)

            acc = 100. * total_correct / total_samples
            avg_loss = total_loss / len(self.train_loader)
            print(f"[{self.phase}] Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

        self.save_model()

    def save_model(self):
        checkpoint_dir = f"checkpoints/{self.dataset_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{checkpoint_dir}/model_{self.phase}.pth")

        # save fc_hidden activations
        self.model.eval()
        fc_inputs, labels, preds = [], [], []
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                fc, logits = self.model(x, extract_fc_input=True)
                fc_inputs.append(fc.cpu())
                labels.append(y.cpu())
                preds.append(logits.argmax(dim=1).cpu())

        torch.save(torch.cat(fc_inputs), f"{checkpoint_dir}/fc_inputs.pt")
        torch.save(torch.cat(labels), f"{checkpoint_dir}/fc_labels.pt")
        torch.save(torch.cat(preds), f"{checkpoint_dir}/fc_preds.pt")
        torch.save(self.model.classifier.weight.data.cpu(), f"{checkpoint_dir}/classifier_weight.pt")
        torch.save(self.model.classifier.bias.data.cpu(), f"{checkpoint_dir}/classifier_bias.pt")


def GurobiFlipFinalLayer(dataset_name, misclassification_count=1, tol=5e-6):
    X = torch.load(f"checkpoints/{dataset_name}/fc_inputs.pt").numpy()
    labels = torch.load(f"checkpoints/{dataset_name}/fc_labels.pt").numpy()
    preds = torch.load(f"checkpoints/{dataset_name}/fc_preds.pt").numpy()
    W = torch.load(f"checkpoints/{dataset_name}/classifier_weight.pt").numpy()
    b = torch.load(f"checkpoints/{dataset_name}/classifier_bias.pt").numpy()

    Z_target = X @ W.T + b

    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", timeLimit)

    W_offset = model.addVars(*W.shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W_offset")
    b_offset = model.addVars(W.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b_offset")
    misclassified_flags = model.addVars(X.shape[0], vtype=GRB.BINARY, name="misclassified_flags")

    for i in range(X.shape[0]):
        correct = int(np.argmax(Z_target[i]))
        z = model.addVars(W.shape[0], lb=-GRB.INFINITY, name=f"Z2_{i}")
        for j in range(W.shape[0]):
            model.addConstr(z[j] == gp.LinExpr(sum((W[j, k] + W_offset[j, k]) * X[i, k] for k in range(W.shape[1])) + b[j] + b_offset[j]))

        violations = model.addVars(W.shape[0], vtype=GRB.BINARY)
        for j in range(W.shape[0]):
            if j == correct:
                model.addConstr(violations[j] == 0)
            else:
                model.addGenConstrIndicator(violations[j], True, z[correct] <= z[j] - tol)
                model.addGenConstrIndicator(violations[j], False, z[correct] >= z[j] + tol)

        model.addConstr(gp.quicksum(violations[j] for j in range(W.shape[0])) >= misclassified_flags[i])
        model.addConstr(gp.quicksum(violations[j] for j in range(W.shape[0])) <= (W.shape[0] - 1) * misclassified_flags[i])

    model.addConstr(gp.quicksum(misclassified_flags[i] for i in range(X.shape[0])) == misclassification_count)

    absW = model.addVars(*W.shape, lb=0)
    absb = model.addVars(W.shape[0], lb=0)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            model.addConstr(absW[i, j] >= W_offset[i, j])
            model.addConstr(absW[i, j] >= -W_offset[i, j])
        model.addConstr(absb[i] >= b_offset[i])
        model.addConstr(absb[i] >= -b_offset[i])

    model.setObjective(gp.quicksum(absW[i, j] for i in range(W.shape[0]) for j in range(W.shape[1])) + gp.quicksum(absb[i] for i in range(W.shape[0])), GRB.MINIMIZE)

    model.optimize()
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        W_new = W + np.array([[W_offset[i, j].X for j in range(W.shape[1])] for i in range(W.shape[0])])
        b_new = b + np.array([b_offset[i].X for i in range(W.shape[0])])
        return W_new, b_new
    else:
        print("Gurobi failed.")
        return None, None


# Example usage:
# model = SimpleGNNClassifier(in_dim=embedding_dim, hidden_dim=32, out_dim=num_classes)
# raf = RAF("Cora", model, train_loader, test_loader, device)
# raf.train()
# W_new, b_new = GurobiFlipFinalLayer("Cora", misclassification_count=10)
# if W_new is not None:
#     model.classifier.weight.data = torch.tensor(W_new).to(device)
#     model.classifier.bias.data = torch.tensor(b_new).to(device)
#     raf = RAF("Cora", model, train_loader, test_loader, device, num_epochs=100, phase="Resume")
#     raf.train()
