import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np

log_file = "Status.txt"
initial_epoch = 200
resume_epoch = 100
timeLimit = 600


class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(96, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1),
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(10 * 8 * 8, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x.clone().detach(), None
        return x


def GurobiFlip():
    n_samples = 100
    X = torch.load("checkpoints/CIFER10/fc_inputs.pt").numpy()[0:n_samples]
    labels = torch.load("checkpoints/CIFER10/fc_labels.pt").numpy()[0:n_samples]
    pred = torch.load("checkpoints/CIFER10/fc_preds.pt").numpy()[0:n_samples]
    X_full = torch.load("checkpoints/CIFER10/fc_inputs.pt").numpy()
    labels_full = torch.load("checkpoints/CIFER10/fc_labels.pt").numpy()
    pred_full = torch.load("checkpoints/CIFER10/fc_preds.pt").numpy()

    # X = torch.load("checkpoints/CIFER10/fc_inputs.pt").numpy()
    # labels = torch.load("checkpoints/CIFER10/fc_labels.pt").numpy()
    # pred = torch.load("checkpoints/CIFER10/fc_preds.pt").numpy()

    W1 = torch.load("checkpoints/CIFER10/fc_hidden_weight.pt").numpy()
    b1 = torch.load("checkpoints/CIFER10/fc_hidden_bias.pt").numpy()
    W2 = torch.load("checkpoints/CIFER10/classifier_weight.pt").numpy()
    b2 = torch.load("checkpoints/CIFER10/classifier_bias.pt").numpy()

    Z1 = np.maximum(0, X @ W1.T + b1)
    Z2_target = Z1 @ W2.T + b2  
    preds_Z2 = np.argmax(Z2_target, axis=1)

    print("Mismatch: ", sum(preds_Z2 != pred))
    print("Size of X:", X.shape)


    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    model = gp.Model()
    # model.setParam("OutputFlag", 1)

    W2_offset = model.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
    b2_offset = model.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")

    Z2_list = []
    max_min_diff = []

    misclassified_flags = model.addVars(n_samples, vtype=GRB.BINARY, name="misclassified_flags")
    epsilon = 1e-6
    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        label_min = int(np.argmin(Z2_target[s]))
        A1_fixed = Z1[s]

        Z2 = model.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b2[j] + b2_offset[j]
            model.addConstr(Z2[j] == expr)

        violations = model.addVars(l2_size, vtype=GRB.BINARY, name=f"violations_{s}")
        for k in range(l2_size):
            if k != label_max:
                model.addConstr((violations[k] == 1) >> (Z2[label_max] <= Z2[k] - epsilon), name=f"violation_1flip_{s}_{k}")
                model.addConstr((violations[k] == 0) >> (Z2[label_max] >= Z2[k] + epsilon), name=f"violation_0flip_{s}_{k}")
            else:
                model.addConstr(violations[k] == 0, name=f"violation_0_{s}_{k}")

        model.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) >= misclassified_flags[s])
        model.addConstr(gp.quicksum(violations[k] for k in range(l2_size)) <= (l2_size - 1) * misclassified_flags[s])
    
    model.addConstr(gp.quicksum(misclassified_flags[s] for s in range(n_samples)) == 1, name="exactly_one_misclassified")

    
    abs_W2 = model.addVars(*W2.shape, lb=0, name="abs_W2")
    abs_b2 = model.addVars(l2_size, lb=0, name="abs_b2")

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            model.addConstr(abs_W2[i, j] >= W2_offset[i, j])
            model.addConstr(abs_W2[i, j] >= -W2_offset[i, j])
    for i in range(l2_size):
        model.addConstr(abs_b2[i] >= b2_offset[i])
        model.addConstr(abs_b2[i] >= -b2_offset[i])
    
    objective = (
        gp.quicksum(abs_W2[i, j] for i in range(W2.shape[0]) for j in range(W2.shape[1])) +
        gp.quicksum(abs_b2[i] for i in range(l2_size))
    )
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

        print(f"Misclassified: {misclassified}, out of {n_samples}")
        print("Average Cross Entropy loss (Z2 vs labels):", ce_loss_target / n_samples)
        print("Average Cross Entropy loss (z2 vs labels):", ce_loss_pred / n_samples)
        """ Full dataset evaluation """
        Z1_full = np.maximum(0, X_full @ W1.T + b1)
        Z2_target_full = Z1_full @ W2.T + b2  
        for i in range(len(X_full)):
            x = X_full[i]
            label = int(np.argmax(Z2_target_full[i]))
            a1 = relu(W1 @ x + b1)
            z2 = W2_new @ a1 + b2_new
            pred = np.argmax(z2)

            predictions.append(pred)
            true_labels.append(label)
            if pred != label:
                # print(f"Sample {i} misclassified: true={label}, pred={pred}")
                misclassified += 1

            pred_probs = softmax(z2)
            target_probs = softmax(Z2_target_full[i])
            ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
            ce_loss_target += -np.log(target_probs[label] + 1e-12)

        print(f"Misclassified On full dataset: {misclassified} out of {len(X_full)}")
        print("Average Cross Entropy loss (Z2 vs labels):", ce_loss_target / n_samples)
        print("Average Cross Entropy loss (z2 vs labels):", ce_loss_pred / n_samples)

        return
        with open(log_file, "a") as f:
            f.write("------------------------\n")
            f.write("Training With Gurobi Edit\n")
            f.write("------------------------\n")
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NIN(num_classes=10).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        checkpoint = torch.load("./checkpoints/CIFER10/full_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        new_W = torch.tensor(W2_new).to(model.classifier.weight.device)
        new_b = torch.tensor(b2_new).to(model.classifier.bias.device)
        with torch.no_grad():
            model.classifier.weight.copy_(new_W)
            model.classifier.bias.copy_(new_b)

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
        for epoch in range(resume_epoch):
            model.train()
            correct = 0
            total = 0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/200"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            scheduler.step()
            print(f"Train Accuracy after Epoch {epoch+1}: {100. * correct / total:.2f}%")
            with open(log_file, "a") as f:
                f.write(f"Train Accuracy after Epoch {epoch+1}: {100. * correct / total:.2f}%\n")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"Test Accuracy after Gurobi edit: {100. * correct / total:.2f}%")
        with open(log_file, "a") as f:
            f.write(f"Test Accuracy after Gurobi edit: {100. * correct / total:.2f}%\n")
    else:
        print("No solution found.")

if __name__ == "__main__":
    GurobiFlip()    