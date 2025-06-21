import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader, Subset
from tqdm import tqdm
import os
import sys
import time

import numpy as np

from medmnist import PathMNIST

from CNNetworks import NIN_MNIST, NIN_CIFAR10, NIN_SVHN, NIN_EMNIST, NIN, VGG


timeLimit = 3600

class RAB:
    def __init__(self, dataset_name, model, train_loader, val_loader, device, num_epochs=200, resume_epochs=100, batch_size=64, learning_rate=0.01, optimizer_type='SGD', phase = "Train", run_id=0, start_experiment=False):
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        self.log_file = f"Stats/RAB_CrossVal_All/{self.dataset_name}_log.csv"
        
        if start_experiment == True:
            with open(self.log_file, "w") as f:
                f.write("Run,Phase,Epoch,Train_loss,Train_acc\n")

    def train(self, early_stopping_patience=10, min_delta=1e-5):
        self.model.train()
        loss = -1
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.num_epochs+self.resume_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            # for i, (inputs, labels) in enumerate(self.train_loader):
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
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')

            with open(self.log_file, "a") as f:
                f.write(f"{self.run_id},{self.phase},{epoch+1},{avg_train_loss},{train_accuracy}\n")
                        

            if best_train_loss - avg_train_loss > min_delta:
                best_train_loss = avg_train_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs based on training loss.")
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
            for inputs, labels in self.val_loader:
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
            f.write(f"{self.run_id},{self.phase}_Test,-1,{avg_loss},{accuracy}\n")
        return accuracy

    def save_model(self, loss, save_suffix=""):
        checkpoint_dir = f"./checkpoints/{self.dataset_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(self.model.fc_hidden.weight.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_fc_hidden_weight{save_suffix}.pt")
        torch.save(self.model.fc_hidden.bias.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_fc_hidden_bias{save_suffix}.pt")
        torch.save(self.model.classifier.weight.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_classifier_weight{save_suffix}.pt")
        torch.save(self.model.classifier.bias.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_classifier_bias{save_suffix}.pt")
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss.item()
        }, f"{checkpoint_dir}/Run{self.run_id}_full_checkpoint{save_suffix}.pth")

        # self.save_fc_inputs("Train", save_suffix=save_suffix)
        # self.save_fc_inputs("Val", save_suffix=save_suffix)

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
    
    def delete_fc_inputs(self):
        checkpoint_dir_input = f"./checkpoints_inputs/{self.dataset_name}"
        
        try:
            os.remove(f"{checkpoint_dir_input}/fc_inputs_train.pt")
            os.remove(f"{checkpoint_dir_input}/fc_labels_train.pt")
            os.remove(f"{checkpoint_dir_input}/fc_preds_train.pt")
            os.remove(f"{checkpoint_dir_input}/fc_inputs_val.pt")
            os.remove(f"{checkpoint_dir_input}/fc_labels_val.pt")
            os.remove(f"{checkpoint_dir_input}/fc_preds_val.pt")
        except FileNotFoundError:
            print(f"Files for not found for detele.")

    def run(self):
        start_time = time.time()
        self.train(early_stopping_patience=10)
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
    os.makedirs("Stats/RAB_CrossVal_All", exist_ok=True)
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
    
    
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_size = len(train_dataset)
    val_size = len(test_dataset)
    total_size = train_size + val_size
    total_run = 5
    for i in range(1, total_run + 1):
        start_experiment = True if i == 1 else False

        rng = np.random.default_rng(seed=i*42)
        all_indices = rng.permutation(total_size)

        new_train_indices = all_indices[:train_size]
        new_val_indices = all_indices[train_size:]
        
        train_subset = Subset(full_dataset, new_train_indices)
        val_subset = Subset(full_dataset, new_val_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        
        rab = RAB(dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=0.01, optimizer_type=optimize, phase="Train", run_id=i, start_experiment=start_experiment)
        try:
            rab.run()
        except Exception as e:
            print(f"Error during training: {e}")
            total_run += 1
            continue


