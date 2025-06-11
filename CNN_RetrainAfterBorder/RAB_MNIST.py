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

log_file = "Status_MNIST.txt"
initial_epoch = 400
resume_epoch = 200
timeLimit = 600


class NIN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN_MNIST, self).__init__()
        def nin_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            )
        self.features = nn.Sequential(
            nin_block(1, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nin_block(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(32*14*14, 32)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

def TrainAndSave(resume=False):
    if resume:
        with open(log_file, "a") as f:
            f.write("------------------------\n")
            f.write(f"Training Resumed Without Gurobi Edit\n")
            f.write("------------------------\n")

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    model = NIN_MNIST(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    checkpoint_dir = "./checkpoints/MNIST"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if resume:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "full_checkpoint.pth"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        save_suffix = "_resume"
    else:
        start_epoch = 0
        save_suffix = ""

    end_epoch = initial_epoch if not resume else initial_epoch + resume_epoch
    for epoch in range(start_epoch, end_epoch):
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
        acc = 100. * correct / total
        print(f"Train Accuracy after Epoch {epoch+1}: {acc:.2f}%")
        print(f"Loss: {loss.item():.4f}")

        with open(log_file, "a") as f:
            f.write(f"Train Accuracy after Epoch {epoch+1}: {acc:.2f}%\n")
            f.write(f"Loss: {loss.item():.4f}\n")
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
    test_acc = 100. * correct / total
    print(f"Test Accuracy after training: {test_acc:.2f}%")
    with open(log_file, "a") as f:
        f.write(f"Test Accuracy after training: {test_acc:.2f}%\n")

    torch.save(model.fc_hidden.weight.data.clone(), f"{checkpoint_dir}/fc_hidden_weight{save_suffix}.pt")
    torch.save(model.fc_hidden.bias.data.clone(), f"{checkpoint_dir}/fc_hidden_bias{save_suffix}.pt")
    torch.save(model.classifier.weight.data.clone(), f"{checkpoint_dir}/classifier_weight{save_suffix}.pt")
    torch.save(model.classifier.bias.data.clone(), f"{checkpoint_dir}/classifier_bias{save_suffix}.pt")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item()
    }, f"{checkpoint_dir}/full_checkpoint{save_suffix}.pth")

    model.eval()
    X_fc_input = []
    Y_true = []
    Y_pred = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            fc_input, _ = model(inputs, extract_fc_input=True)
            logits = model.classifier(torch.relu(model.fc_hidden(fc_input)))
            preds = torch.argmax(logits, dim=1)
            X_fc_input.append(fc_input.cpu())
            Y_true.append(labels.cpu())
            Y_pred.append(preds.cpu())

    X_fc_input = torch.cat(X_fc_input, dim=0)
    Y_true = torch.cat(Y_true, dim=0)
    Y_pred = torch.cat(Y_pred, dim=0)

    torch.save(X_fc_input, f"{checkpoint_dir}/fc_inputs{save_suffix}.pt")
    torch.save(Y_true, f"{checkpoint_dir}/fc_labels{save_suffix}.pt")
    torch.save(Y_pred, f"{checkpoint_dir}/fc_preds{save_suffix}.pt")

    print(f"Training and saving completed in {time.time() - t0:.2f} seconds.")

def GurobiBorder():
    X = torch.load("checkpoints/MNIST/fc_inputs.pt").numpy()
    labels = torch.load("checkpoints/MNIST/fc_labels.pt").numpy()
    pred = torch.load("checkpoints/MNIST/fc_preds.pt").numpy()

    W1 = torch.load("checkpoints/MNIST/fc_hidden_weight.pt").cpu().numpy()
    b1 = torch.load("checkpoints/MNIST/fc_hidden_bias.pt").cpu().numpy()
    W2 = torch.load("checkpoints/MNIST/classifier_weight.pt").cpu().numpy()
    b2 = torch.load("checkpoints/MNIST/classifier_bias.pt").cpu().numpy()

    Z1 = np.maximum(0, X @ W1.T + b1)
    Z2_target = Z1 @ W2.T + b2  
    preds_Z2 = np.argmax(Z2_target, axis=1)

    print("Mismatch: ", sum(pred != preds_Z2))
    print("Size of X:", X.shape)


    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    model = gp.Model()
    model.setParam("OutputFlag", 1)

    W2_offset = model.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
    b2_offset = model.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")

    Z2_list = []
    max_min_diff = []

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

        for k in range(l2_size):
            if k != label_max:
                model.addConstr(Z2[label_max] >= Z2[k] + 3e-5, f"Z2_max_{s}_{k}")

        Z2_list.append(Z2)
        max_min_diff.append(Z2[label_max] - Z2[label_min])

    objective = gp.quicksum(max_min_diff)
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

        print(f"Misclassified: {misclassified}")
        print("Average Cross Entropy loss (Z2 vs labels):", ce_loss_target / n_samples)
        print("Average Cross Entropy loss (z2 vs labels):", ce_loss_pred / n_samples)

        with open(log_file, "a") as f:
            f.write("------------------------\n")
            f.write("Training With Gurobi Edit\n")
            f.write("------------------------\n")
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NIN_MNIST(num_classes=10).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        checkpoint = torch.load("./checkpoints/MNIST/full_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        new_W = torch.tensor(W2_new).to(model.classifier.weight.device)
        new_b = torch.tensor(b2_new).to(model.classifier.bias.device)
        with torch.no_grad():
            model.classifier.weight.copy_(new_W)
            model.classifier.bias.copy_(new_b)

        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
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
    # TrainAndSave()
    GurobiBorder()
    # TrainAndSave(resume=True)
    