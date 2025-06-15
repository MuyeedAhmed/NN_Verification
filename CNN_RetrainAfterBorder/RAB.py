import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import sys
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from medmnist import PathMNIST, BloodMNIST, OrganAMNIST, OrganAMNIST, OrganSMNIST

from CNNetworks import NIN_MNIST, NIN_CIFAR10, NIN_SVHN, NIN_EMNIST, NIN, VGG


timeLimit = 3600

class RAB:
    def __init__(self, dataset_name, model, train_loader, test_loader, device, num_epochs=200, resume_epochs=100, batch_size=64, learning_rate=0.01, optimizer_type='SGD', phase = "Train"):
        self.dataset_name = dataset_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.resume_epochs = resume_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.phase = phase
        if optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif optimizer_type == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        else:
            raise ValueError("Unsupported optimizer type. Use 'SGD' or 'Adam'.")
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.criterion = nn.CrossEntropyLoss()
        self.log_file = f"Stats/{self.dataset_name}_log.csv"
        
        if phase == "Train":
            with open(self.log_file, "w") as f:
                f.write("Phase,Epoch,Loss,Accuracy\n")

    def train(self, early_stopping_patience=10, min_delta=1e-5):
        self.model.train()
        loss = -1
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.num_epochs+self.resume_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            # for i, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels_for_loss).sum().item()

            self.scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100. * correct / total

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            with open(self.log_file, "a") as f:
                f.write(f"{self.phase},{epoch+1},{avg_loss:.4f},{accuracy:.2f}\n")

            # Early stopping logic
            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

            if epoch == self.num_epochs - 1:
                if self.phase == "Train":
                    self.save_model(loss, save_suffix="")
                    test_accuracy = self.test()
                self.phase = "ResumeTrain"
        if self.phase == "Train":
            self.save_model(loss, save_suffix="")
        elif self.phase == "GurobiEdit":
            self.save_model(loss, save_suffix="_GurobiEdit")
        elif self.phase == "ResumeTrain":
            self.save_model(loss, save_suffix="_Resume")
        

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.dataset_name == "EMNIST":
                    labels_for_loss = labels - 1
                else:
                    labels_for_loss = labels   
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels_for_loss).sum().item()
                loss = self.criterion(outputs, labels_for_loss)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.test_loader)

        accuracy = 100. * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        with open(self.log_file, "a") as f:
                f.write(f"{self.phase}_Test,-1,{total_loss},{accuracy:.2f}\n")
        return accuracy

    def save_model(self, loss, save_suffix=""):
        checkpoint_dir = f"./checkpoints/{self.dataset_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.model.fc_hidden.weight.data.clone(), f"{checkpoint_dir}/fc_hidden_weight{save_suffix}.pt")
        torch.save(self.model.fc_hidden.bias.data.clone(), f"{checkpoint_dir}/fc_hidden_bias{save_suffix}.pt")
        torch.save(self.model.classifier.weight.data.clone(), f"{checkpoint_dir}/classifier_weight{save_suffix}.pt")
        torch.save(self.model.classifier.bias.data.clone(), f"{checkpoint_dir}/classifier_bias{save_suffix}.pt")
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss.item()
        }, f"{checkpoint_dir}/full_checkpoint{save_suffix}.pth")

        self.model.eval()
        X_fc_input = []
        Y_true = []
        Y_pred = []
        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                fc_input, _ = self.model(inputs, extract_fc_input=True)
                logits = self.model.classifier(torch.relu(self.model.fc_hidden(fc_input)))
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

    def run(self):
        start_time = time.time()
        self.train()
        accuracy = self.test()

def GurobiBorder(dataset_name, n=-1, tol = 5e-6):
    X_full = torch.load(f"checkpoints/{dataset_name}/fc_inputs.pt").numpy()
    labels_full = torch.load(f"checkpoints/{dataset_name}/fc_labels.pt").numpy()
    pred_full = torch.load(f"checkpoints/{dataset_name}/fc_preds.pt").numpy()
    X_full_size = X_full.shape[0]
    if n == -1:
        X = X_full
        labels = labels_full
        pred = pred_full
    else:
        X = X_full[0:n]
        labels = labels_full[0:n]
        pred = pred_full[0:n]

    W1 = torch.load(f"checkpoints/{dataset_name}/fc_hidden_weight.pt", map_location=torch.device('cpu')).cpu().numpy()
    b1 = torch.load(f"checkpoints/{dataset_name}/fc_hidden_bias.pt", map_location=torch.device('cpu')).cpu().numpy()
    W2 = torch.load(f"checkpoints/{dataset_name}/classifier_weight.pt", map_location=torch.device('cpu')).cpu().numpy()
    b2 = torch.load(f"checkpoints/{dataset_name}/classifier_bias.pt", map_location=torch.device('cpu')).cpu().numpy()

    Z1 = np.maximum(0, X @ W1.T + b1)
    Z2_target = Z1 @ W2.T + b2

    print("Mismatch: ", sum(pred != np.argmax(Z2_target, axis=1)))
    print("Size of X:", X.shape)
    print("Size of W2:", W2.shape)
    print("Size of b2:", b2.shape)

    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    model_g = gp.Model()
    # model_g.setParam("OutputFlag", 1)

    W2_offset = model_g.addVars(*W2.shape, lb=-GRB.INFINITY, name="W2_offset")
    b2_offset = model_g.addVars(l2_size, lb=-GRB.INFINITY, name="b2_offset")

    Z2_list = []
    max_min_diff = []

    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        label_min = int(np.argmin(Z2_target[s]))
        A1_fixed = Z1[s]

        Z2 = model_g.addVars(l2_size, lb=-GRB.INFINITY, name=f"Z2_{s}")
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b2[j] + b2_offset[j]
            model_g.addConstr(Z2[j] == expr)

        for k in range(l2_size):
            if k != label_max:
                model_g.addConstr(Z2[label_max] >= Z2[k] + tol, f"Z2_max_{s}_{k}")

        Z2_list.append(Z2)
        max_min_diff.append(Z2[label_max] - Z2[label_min])

    objective = gp.quicksum(max_min_diff)
    model_g.setObjective(objective, GRB.MINIMIZE)
    model_g.addConstr(objective >= 0, "ObjectiveLowerBound")
    model_g.setParam('TimeLimit', timeLimit)
    model_g.optimize()

    if model_g.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and model_g.SolCount > 0:
        W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off
        def relu(x): return np.maximum(0, x)
        def softmax(logits):
            e = np.exp(logits - np.max(logits))
            return e / np.sum(e)

        misclassified = 0
        ce_loss_target = 0
        ce_loss_pred = 0
        # predictions, true_labels = [], []

        for i in range(n_samples):
            x = X[i]
            label = int(np.argmax(Z2_target[i]))
            a1 = relu(W1 @ x + b1)
            z2 = W2_new @ a1 + b2_new
            pred = np.argmax(z2)

            # predictions.append(pred)
            # true_labels.append(label)
            if pred != label:
                print(f"Sample {i} misclassified: true={label}, pred={pred}")
                misclassified += 1

            pred_probs = softmax(z2)
            target_probs = softmax(Z2_target[i])
            ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
            ce_loss_target += -np.log(target_probs[label] + 1e-12)
        if misclassified > 0:
            with open(f"Stats/{dataset_name}_gurobi_log_tol.csv", "a") as f:
                f.write(f"Tol:{tol}\nMisclassified: {misclassified}\n")
            GurobiBorder(dataset_name, n=n, tol=tol+5e-6)
        with open(f"Stats/{dataset_name}_gurobi_log.csv", "w") as f:
            f.write("-------Weight/Bias Offsets-------\n")
            f.write(f"W2 offsets: {np.sum(np.abs(W2_off))}\n")
            f.write(f"b2 offsets: {np.sum(np.abs(b2_off))}\n")
            f.write(f"Objective value: {model_g.ObjVal}\n")
            f.write("------------------------------------\n\n")
            f.write(f"Misclassified: {misclassified}\n")
            f.write("Average Cross Entropy loss (Z2 vs labels): " + str(ce_loss_target / n_samples) + "\n")
            f.write("Average Cross Entropy loss (z2 vs labels): " + str(ce_loss_pred / n_samples) + "\n")
        
        if n != -1:
            Z1_full = np.maximum(0, X_full @ W1.T + b1)
            Z2_full_target = Z1_full @ W2.T + b2
            
            for i in range(X_full_size):
                a1 = relu(W1 @ X_full[i] + b1)
                z2 = W2_new @ a1 + b2_new
                pred = np.argmax(z2)
                label = int(np.argmax(Z2_full_target[i]))
                if pred != label:
                    misclassified += 1
                pred_probs = softmax(z2)
                target_probs = softmax(Z2_full_target[i])
                ce_loss_pred += -np.log(pred_probs[label] + 1e-12)
                ce_loss_target += -np.log(target_probs[label] + 1e-12)
            ce_loss_target /= X_full_size
            ce_loss_pred /= X_full_size
            with open(f"Stats/{dataset_name}_gurobi_log.csv", "a") as f:
                f.write("\n\n-------Full Dataset Statistics-------\n")
                f.write(f"Total samples in full dataset: {X_full_size}\n")
                f.write(f"Total misclassified samples in full dataset: {misclassified}\n")
                f.write(f"Average Cross Entropy loss (Z2 vs labels): {ce_loss_target}\n")
                f.write(f"Average Cross Entropy loss (z2 vs labels): {ce_loss_pred}\n")
        
        W2_new = W2 + W2_off
        b2_new = b2 + b2_off

        return [W2_new, b2_new]
    else:
        print("No solution found.")
        return None
        
        
class WrapOneHotEncoding(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label = label.argmax().item()
        return image, label

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    os.makedirs("Stats", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    initEpoch = 200
    G_epoch = 100
    n_samples_gurobi = -1
    optimize = "Adam"
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "MNIST"

    train_loader = None
    test_loader = None
    model = None

    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


        model = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        # model = NIN_CIFAR10(num_classes=10).to(device)
        # model_g = NIN_CIFAR10(num_classes=10).to(device)
        model = VGG(num_classes=10).to(device)
        model_g = VGG(num_classes=10).to(device)

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "KMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    
    elif dataset_name == "EMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        optimize = "SGD"
        model = NIN_EMNIST(num_classes=26).to(device)
        model_g = NIN_EMNIST(num_classes=26).to(device)
    
    elif dataset_name == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = VGG(num_classes=10).to(device)
        model_g = VGG(num_classes=10).to(device)

    elif dataset_name == "PathMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        train_raw = PathMNIST(split='train', download=True, transform=transform)
        test_raw = PathMNIST(split='test', download=True, transform=transform)

        train_dataset = WrapOneHotEncoding(train_raw)
        test_dataset = WrapOneHotEncoding(test_raw)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = VGG(num_classes=9).to(device)
        model_g = VGG(num_classes=9).to(device)
    
    elif dataset_name == "BloodMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        train_raw = BloodMNIST(split='train', download=True, transform=transform)
        test_raw = BloodMNIST(split='test', download=True, transform=transform)

        train_dataset = WrapOneHotEncoding(train_raw)
        test_dataset = WrapOneHotEncoding(test_raw)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = VGG(num_classes=8).to(device)
        model_g = VGG(num_classes=8).to(device)
    
    elif dataset_name == "OrganAMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_raw = OrganAMNIST(split='train', download=True, transform=transform)
        test_raw = OrganAMNIST(split='test', download=True, transform=transform)

        train_dataset = WrapOneHotEncoding(train_raw)
        test_dataset = WrapOneHotEncoding(test_raw)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = NIN_MNIST(num_classes=11).to(device)
        model_g = NIN_MNIST(num_classes=11).to(device)

    if os.path.exists(f"./checkpoints/{dataset_name}/full_checkpoint.pth") == False:
        rab = RAB(dataset_name, model, train_loader, test_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=0.01, optimizer_type=optimize, phase="Train")
        rab.run()

    Gurobi_output = GurobiBorder(dataset_name, n=n_samples_gurobi)
    if Gurobi_output is None:
        print("Gurobi did not find a solution.")
        sys.exit(1)
    W2_new, b2_new = Gurobi_output

    rab_g = RAB(dataset_name, model_g, train_loader, test_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=64, learning_rate=0.01, optimizer_type=optimize, phase="GurobiEdit")

    if device.type == 'cuda':
        checkpoint = torch.load(f"./checkpoints/{dataset_name}/full_checkpoint.pth")
    else:
        checkpoint = torch.load(f"./checkpoints/{dataset_name}/full_checkpoint.pth", map_location=torch.device('cpu'))
    rab_g.model.load_state_dict(checkpoint['model_state_dict'])
    rab_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    rab_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    new_W = torch.tensor(W2_new).to(model_g.classifier.weight.device)
    new_b = torch.tensor(b2_new).to(model_g.classifier.bias.device)
    with torch.no_grad():
        rab_g.model.classifier.weight.copy_(new_W)
        rab_g.model.classifier.bias.copy_(new_b)
    rab_g.run()