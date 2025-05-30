import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x, extract_fc_input=False):
        for i in range(12):
            x = self.features[i](x)
        if extract_fc_input:
            fc_input = x.clone().detach()
        x = self.features[12](x)
        x = self.features[13](x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x, fc_input
        return x

# === Step 1: Train and Save Initial Model ===
def step1_train_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    model = NIN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(30):
        model.train()
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
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

    model.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"Test Accuracy after initial training: {100. * correct / total:.2f}%")

    os.makedirs("./checkpoints/MNIST", exist_ok=True)
    for module in model.features:
        if isinstance(module, nn.Conv2d) and module.out_channels == 10:
            final_conv = module
            break

    torch.save(final_conv.weight.data.clone(), "./checkpoints/MNIST/last_weight_original.pt")
    torch.save(final_conv.bias.data.clone(), "./checkpoints/MNIST/last_bias_original.pt")

    model.eval()
    X_fc = []
    Y_true = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _, fc_input = model(inputs, extract_fc_input=True)
        X_fc.append(fc_input.cpu())
        Y_true.append(labels.cpu())
    X_fc = torch.cat(X_fc, dim=0).view(len(test_dataset), -1)
    torch.save(X_fc, "./checkpoints/MNIST/fc_inputs.pt")
    torch.save(torch.cat(Y_true, dim=0), "./checkpoints/MNIST/fc_labels.pt")

def step2_optimize_final_layer():
    X_fc = torch.load("./checkpoints/MNIST/fc_inputs.pt").numpy()
    labels = torch.load("./checkpoints/MNIST/fc_labels.pt").numpy()
    W = torch.load("./checkpoints/MNIST/last_weight_original.pt").view(10, -1).numpy()
    b = torch.load("./checkpoints/MNIST/last_bias_original.pt").numpy()

    Z2_target = X_fc @ W.T + b
    n_samples = len(X_fc)
    l1_size = W.shape[1]
    l2_size = W.shape[0]

    model_gp = gp.Model()
    model_gp.setParam("OutputFlag", 1)
    model_gp.setParam("TimeLimit", 18000)

    W2_offset = model_gp.addVars(*W.shape, lb=-GRB.INFINITY, name="W2_offset")
    b2_offset = model_gp.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")

    max_min_diff = []
    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        label_min = int(np.argmin(Z2_target[s]))
        A1_fixed = X_fc[s]

        Z2 = model_gp.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b[j] + b2_offset[j]
            model_gp.addConstr(Z2[j] == expr)

        for k in range(l2_size):
            if k != label_max:
                model_gp.addConstr(Z2[label_max] >= Z2[k] + 3e-5, f"Z2_max_{s}_{k}")

        max_min_diff.append(Z2[label_max] - Z2[label_min])

    objective = gp.quicksum(max_min_diff)
    model_gp.setObjective(objective, GRB.MINIMIZE)
    model_gp.addConstr(objective >= 0, "ObjectiveLowerBound")
    model_gp.optimize()

    if model_gp.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and model_gp.SolCount > 0:
        W2_off = np.array([[W2_offset[i, j].X for j in range(W.shape[1])] for i in range(W.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])
        W2_new = W + W2_off
        b2_new = b + b2_off

        torch.save(torch.tensor(W2_new), "./checkpoints/MNIST/last_weight_optimized.pt")
        torch.save(torch.tensor(b2_new), "./checkpoints/MNIST/last_bias_optimized.pt")
# === Step 3: Fine-Tune with Modified Weights ===
def step3_finetune_modified_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    model = NIN(num_classes=10).to(device)
    model.load_state_dict(torch.load("./checkpoints/MNIST/nin_finetuned.pth", map_location=device), strict=False)

    for module in model.features:
        if isinstance(module, nn.Conv2d) and module.out_channels == 10:
            final_conv = module
            break

    final_conv.weight.data.copy_(torch.load("./checkpoints/MNIST/last_weight_optimized.pt").view_as(final_conv.weight))
    final_conv.bias.data.copy_(torch.load("./checkpoints/MNIST/last_bias_optimized.pt"))

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/20"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {100.*correct/total:.2f}%")

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
    print(f"Test Accuracy after fine-tuning: {100. * correct / total:.2f}%")

    torch.save(model.state_dict(), "./checkpoints/MNIST/nin_finetuned.pth")

# === Run the Full Pipeline ===
if __name__ == "__main__":
    step1_train_and_save()
    # step2_optimize_final_layer()
    # step3_finetune_modified_model()
