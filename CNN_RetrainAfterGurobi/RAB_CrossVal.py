import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
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
    def __init__(self, dataset_name, model, train_loader, val_loader, test_loader, device, num_epochs=200, resume_epochs=100, batch_size=64, learning_rate=0.01, optimizer_type='SGD', phase = "Train", run_id=0):
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        self.log_file = f"Stats_RAB_CrossVal/{self.dataset_name}_log.csv"
        
        if phase == "Train" and self.run_id == 1:
            with open(self.log_file, "w") as f:
                f.write("Run,Phase,Epoch,Train_loss,Train_acc,Val_loss,Val_acc\n")

    def train(self, early_stopping_patience=15, min_delta=1e-5, stopper="Val"):
        self.model.train()
        loss = -1
        best_val_loss = float('inf')
        best_train_loss = float('inf')
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
            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100. * correct / total

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels_for_loss)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels_for_loss).sum().item()
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = 100. * val_correct / val_total
            self.model.train()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

            with open(self.log_file, "a") as f:
                f.write(f"{self.run_id},{self.phase},{epoch+1},{avg_train_loss},{train_accuracy},"
                        f"{avg_val_loss},{val_accuracy}\n")

            if stopper == "Train":
                if best_train_loss - avg_train_loss > min_delta:
                    best_train_loss = avg_train_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs based on training loss.")
                    break
            elif stopper == "Val":
                if best_val_loss - avg_val_loss > min_delta:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs based on validation loss.")
                    break

            if epoch == self.num_epochs - 1:
                if self.phase == "Train":
                    self.save_model(loss, save_suffix="")
                    test_accuracy = self.test()
                self.phase = "ResumeTrain"
        if self.phase == "Train":
            self.save_model(loss, save_suffix="")
        elif self.phase == "GurobiEdit":
            self.save_model(loss, save_suffix="_GE_RAB")
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
                labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)

                batch_size = inputs.size(0)
                total += batch_size
                total_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels_for_loss).sum().item()
                
        avg_loss = total_loss / total

        accuracy = 100. * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        with open(self.log_file, "a") as f:
            f.write(f"{self.run_id},{self.phase}_Test,-1,{avg_loss},{accuracy},-1,-1\n")
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

        self.save_fc_inputs("Train", save_suffix=save_suffix)
        self.save_fc_inputs("Val", save_suffix=save_suffix)

    def save_fc_inputs(self, dataset_type, save_suffix=""):
        checkpoint_dir_input = f"./checkpoints_inputs/{self.dataset_name}"
        os.makedirs(checkpoint_dir_input, exist_ok=True)
        self.model.eval()
        X_fc_input = []
        Y_true = []
        Y_pred = []
        if dataset_type == "Train":
            loader = self.train_loader
            save_suffix = "_train" + save_suffix
        elif dataset_type == "Val":
            loader = self.val_loader
            save_suffix = "_val" + save_suffix

        with torch.no_grad():
            for inputs, labels in loader:
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

        torch.save(X_fc_input, f"{checkpoint_dir_input}/fc_inputs{save_suffix}.pt")
        torch.save(Y_true, f"{checkpoint_dir_input}/fc_labels{save_suffix}.pt")
        torch.save(Y_pred, f"{checkpoint_dir_input}/fc_preds{save_suffix}.pt")

    def run(self, stopper="Val"):
        start_time = time.time()
        self.train(stopper=stopper)
        accuracy = self.test()
    
    def evaluate(self, dataset_type):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        if dataset_type == "Train":
            loader = self.train_loader
        elif dataset_type == "Val":
            loader = self.val_loader
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)
                batch_size = inputs.size(0)

                total_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels_for_loss).sum().item()
                total += batch_size

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy


def GurobiBorder(dataset_name, store_file_name, run_id, n=-1, tol = 1e-5):
    X_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_train.pt").numpy()
    labels_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_train.pt").numpy()
    pred_full = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_train.pt").numpy()
    X_full_size = X_full.shape[0]
    if n == -1:
        X = X_full
        labels_gt = labels_full
        pred_checkpoint = pred_full
    else:
        X = X_full[0:n]
        labels_gt = labels_full[0:n]
        pred_checkpoint = pred_full[0:n]

    W1 = torch.load(f"checkpoints/{dataset_name}/fc_hidden_weight.pt", map_location=torch.device('cpu')).numpy()
    b1 = torch.load(f"checkpoints/{dataset_name}/fc_hidden_bias.pt", map_location=torch.device('cpu')).numpy()
    W2 = torch.load(f"checkpoints/{dataset_name}/classifier_weight.pt", map_location=torch.device('cpu')).numpy()
    b2 = torch.load(f"checkpoints/{dataset_name}/classifier_bias.pt", map_location=torch.device('cpu')).numpy()

    Z1 = np.maximum(0, X @ W1.T + b1)
    Z2_target = Z1 @ W2.T + b2
    
    pred_target = np.argmax(Z2_target, axis=1)
    # print(f"W2 dtype: {W2.dtype}, b2 dtype: {b2.dtype}")
    # print("Size of X:", X.shape)
    # print("Size of W2:", W2.shape)
    # print("Size of b2:", b2.shape)
    print("Mismatch: ", sum(pred_checkpoint != pred_target))

    n_samples = len(X)
    l1_size = W2.shape[1]
    l2_size = W2.shape[0]

    gurobi_model = gp.Model()
    W2_offset = gurobi_model.addVars(*W2.shape, lb=-20, ub=20, name="W2_offset")
    b2_offset = gurobi_model.addVars(l2_size, lb=-20, ub=20, name="b2_offset")

    Z2_list = []
    max_min_diff = []

    for s in range(n_samples):
        label_max = int(np.argmax(Z2_target[s]))
        label_min = int(np.argmin(Z2_target[s]))
        A1_fixed = Z1[s]

        Z2 = gurobi_model.addVars(l2_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Z2_{s}")
        for j in range(l2_size):
            expr = gp.LinExpr()
            for i in range(l1_size):
                expr += (W2[j, i] + W2_offset[j, i]) * A1_fixed[i]
            expr += b2[j] + b2_offset[j]
            gurobi_model.addConstr(Z2[j] == expr)

        for k in range(l2_size):
            if k != label_max:
                gurobi_model.addConstr(Z2[label_max] >= Z2[k] + tol, f"Z2_max_{s}_{k}")

        Z2_list.append(Z2)
        max_min_diff.append(Z2[label_max] - Z2[label_min])

    objective = gp.quicksum(max_min_diff)
    gurobi_model.setObjective(objective, GRB.MINIMIZE)
    # gurobi_model.addConstr(objective >= 0, "ObjectiveLowerBound")
    gurobi_model.setParam('TimeLimit', timeLimit)
    gurobi_model.optimize()

    if gurobi_model.status in [GRB.TIME_LIMIT, GRB.OPTIMAL] and gurobi_model.SolCount > 0:
        W2_off = np.array([[W2_offset[i, j].X for j in range(W2.shape[1])] for i in range(W2.shape[0])])
        b2_off = np.array([b2_offset[i].X for i in range(l2_size)])

        W2_new = (W2 + W2_off)
        b2_new = (b2 + b2_off)
        # print(f"W2 dtype: {W2.dtype}, b2 dtype: {b2.dtype}")
        # print(f"W2_new dtype: {W2_new.dtype}, b2_new dtype: {b2_new.dtype}")
        def softmax(x):
            x = x - np.max(x, axis=1, keepdims=True)
            e_x = np.exp(x)
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        A1 = np.maximum(0, X @ W1.T + b1)
        Z2_pred_gurobi = A1 @ W2_new.T + b2_new
        predictions_gurobi = np.argmax(Z2_pred_gurobi, axis=1)
        misclassified_mask = predictions_gurobi != pred_checkpoint
        misclassified = np.sum(misclassified_mask)
        accuracy_gurobi = np.sum(predictions_gurobi == labels_gt) / len(labels_gt) * 100
        
        if misclassified > 0:
            with open(f"Stats_RAB_CrossVal/{dataset_name}_gurobi_log_tol.csv", "a") as f:
                f.write(f"Tol:{tol}\nMisclassified: {misclassified}\n")
            GurobiBorder(dataset_name, store_file_name, run_id, n=n, tol=tol+5e-6)

        print(f"Total misclassified samples: {misclassified}")
        with open(f"Stats_RAB_CrossVal/{dataset_name}_gurobi_log.csv", "a") as f:
            f.write(f"-----\nRun ID: {run_id}\n\n")
            f.write("-------Weight/Bias Offsets-------\n")
            f.write(f"W2 offsets: {np.sum(np.abs(W2_off))}\n")
            f.write(f"b2 offsets: {np.sum(np.abs(b2_off))}\n")
            f.write(f"Objective value: {gurobi_model.ObjVal}\n")
            f.write("------------------------------------\n\n")
            f.write(f"Misclassified: {misclassified}\n")
        
        X_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_inputs_val.pt").numpy()
        labels_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_labels_val.pt").numpy()
        pred_val = torch.load(f"checkpoints_inputs/{dataset_name}/fc_preds_val.pt").numpy()
        Z1_val = np.maximum(0, X_val @ W1.T + b1)
        Z2_val_pred = Z1_val @ W2_new.T + b2_new
        predictions_val = np.argmax(Z2_val_pred, axis=1)
        accuracy_val = np.sum(predictions_val == labels_val) / len(labels_val) * 100
        
        with open(store_file_name, "a") as f:
            f.write(f"{run_id},GurobiComplete,-1,-1,{accuracy_gurobi},-1,{accuracy_val}\n")

        
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
    os.makedirs("Stats_RAB_CrossVal", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    initEpoch = 200
    G_epoch = 100
    n_samples_gurobi = -1
    optimize = "Adam"

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "MNIST"

    train_loader = None
    test_loader = None
    model_t = None

    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        model_t = VGG(num_classes=10).to(device)
        model_g = VGG(num_classes=10).to(device)

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)

    elif dataset_name == "KMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    
    elif dataset_name == "EMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

        optimize = "SGD"
        # Updated n_samples_gurobi
        n_samples_gurobi = 5000
        model_t = NIN_EMNIST(num_classes=26).to(device)
        model_g = NIN_EMNIST(num_classes=26).to(device)
    
    elif dataset_name == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        
        model_t = VGG(num_classes=10).to(device)
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

        model_t = VGG(num_classes=9).to(device)
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

        model_t = VGG(num_classes=8).to(device)
        model_g = VGG(num_classes=8).to(device)
    
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # if os.path.exists(f"./checkpoints/{dataset_name}/full_checkpoint.pth") == False:
    #     rab = RAB(dataset_name, model_t, train_loader, test_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=0.01, optimizer_type=optimize, phase="Train")
    #     rab.run()
    total_run = 5
    for i in range(1, total_run + 1):
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        generator = torch.Generator().manual_seed(i*42)
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        
        rab = RAB(dataset_name, model_t, train_loader, val_loader, test_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=0.01, optimizer_type=optimize, phase="Train", run_id=i)
        rab.run()
        Gurobi_output = GurobiBorder(dataset_name, rab.log_file, i, n=n_samples_gurobi)
        if Gurobi_output is None:
            print("Gurobi did not find a solution.")
            if total_run < 10:
                total_run += 1
            continue
        W2_new, b2_new = Gurobi_output

        rab_after_g = RAB(dataset_name, model_g, train_loader, val_loader, test_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=64, learning_rate=0.01, optimizer_type=optimize, phase="GurobiEdit", run_id=i)

        if device.type == 'cuda':
            checkpoint = torch.load(f"./checkpoints/{dataset_name}/full_checkpoint.pth")
        else:
            checkpoint = torch.load(f"./checkpoints/{dataset_name}/full_checkpoint.pth", map_location=torch.device('cpu'))
        rab_after_g.model.load_state_dict(checkpoint['model_state_dict'])
        rab_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rab_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        new_W = torch.tensor(W2_new).to(model_g.classifier.weight.device)
        new_b = torch.tensor(b2_new).to(model_g.classifier.bias.device)
        with torch.no_grad():
            rab_after_g.model.classifier.weight.copy_(new_W)
            rab_after_g.model.classifier.bias.copy_(new_b)
        train_loss, train_acc = rab_after_g.evaluate("Train")
        val_loss, val_acc = rab_after_g.evaluate("Val")

        with open(rab_after_g.log_file, "a") as f:
            f.write(f"{i},Gurobi_Complete_Eval,-1,{train_loss},{train_acc},{val_loss},{val_acc}\n")

        rab_after_g.run()

