import torch

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
import random
import numpy as np
from TrainModel import TrainModel
from medmnist import PathMNIST

from CNNetworks import NIN_MNIST, NIN_CIFAR10, NIN_SVHN, NIN_EMNIST, NIN, VGG, CNN_USPS, Food101Net, VGG_office31


def GetModel(dataset_name, num_classes=10, device=None):
    if dataset_name == "MNIST":
        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "CIFAR10":
        model_t = VGG(num_classes=10).to(device)
        model_g = VGG(num_classes=10).to(device)
    elif dataset_name == "FashionMNIST":
        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "KMNIST":
        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "EMNIST":
        model_t = NIN_EMNIST(num_classes=26).to(device)
        model_g = NIN_EMNIST(num_classes=26).to(device)
    elif dataset_name == "SVHN":
        model_t = VGG(num_classes=10).to(device)
        model_g = VGG(num_classes=10).to(device)
    elif dataset_name == "PathMNIST":
        model_t = VGG(num_classes=9).to(device)
        model_g = VGG(num_classes=9).to(device)
    elif dataset_name == "Food101":
            model_t = VGG(num_classes=10).to(device)
            model_g = VGG(num_classes=10).to(device)
    elif dataset_name == "USPS":
        model_t = CNN_USPS(num_classes=10).to(device)
        model_g = CNN_USPS(num_classes=10).to(device)
    elif dataset_name == "Caltech101":
        model_t = VGG(num_classes=101).to(device)
        model_g = VGG(num_classes=101).to(device)
    elif dataset_name == "office31":
        model_t = VGG_office31(num_classes=31).to(device)
        model_g = VGG_office31(num_classes=31).to(device)

    return model_t, model_g

def GetDataset(dataset_name, root_dir='./data', device=None):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "KMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    elif dataset_name == "EMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
    
    elif dataset_name == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    elif dataset_name == "PathMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        train_raw = PathMNIST(split='train', download=True, transform=transform)
        test_raw = PathMNIST(split='test', download=True, transform=transform)

        train_dataset = WrapOneHotEncoding(train_raw)
        test_dataset = WrapOneHotEncoding(test_raw)
    
    elif dataset_name == "Food101":
        transform = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor()])
        train_dataset = torchvision.datasets.Food101(root="./data", split="train", download=True, transform=transform)
        test_dataset = torchvision.datasets.Food101(root="./data", split="test", download=True, transform=transform)
        all_classes = train_dataset.classes
        random.seed(42)
        selected_classes = random.sample(all_classes, 10)

        def fast_filter_food101(dataset, selected_classes):
            class_names = dataset.classes
            label_names = [class_names[label] for label in dataset._labels]
            selected_indices = [i for i, name in enumerate(label_names) if name in selected_classes]
            return Subset(dataset, selected_indices)

        class_map = {cls_name: i for i, cls_name in enumerate(selected_classes)}
        train_filtered = fast_filter_food101(train_dataset, selected_classes)
        test_filtered = fast_filter_food101(test_dataset, selected_classes)

        train_dataset = RelabelSubset(train_filtered, class_map, train_dataset)
        test_dataset = RelabelSubset(test_filtered, class_map, test_dataset)


    elif dataset_name == "USPS":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.USPS(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "Caltech101":
        train_dataset, test_dataset = get_loaders_from_folder("./data/caltech-101/101_ObjectCategories", image_size=(64, 64), val_split=0.25)  

    elif dataset_name == "office31":
        train_dataset, test_dataset = get_loaders_from_folder("./data/office31/amazon", image_size=(64, 64), val_split=0.2)

    return train_dataset, test_dataset   

class RelabelSubset(torch.utils.data.Dataset):
    def __init__(self, subset, class_map, orig_dataset):
        self.subset = subset
        self.class_map = class_map
        self.orig_dataset = orig_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        class_name = self.orig_dataset.classes[label]
        new_label = self.class_map[class_name]
        return image, new_label

class WrapOneHotEncoding(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label = label.argmax().item()
        return image, label

    def __len__(self):
        return len(self.dataset)

def get_loaders_from_folder(root_dir, image_size=(224, 224), val_split=0.2, seed=42):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    full_dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    return train_dataset, val_dataset


if __name__ == "__main__":
    os.makedirs("Stats/RAB_CrossVal_All", exist_ok=True)
    os.makedirs("Stats/RAF_CrossVal_All", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    initEpoch = 300
    G_epoch = 100
    optimize = "Adam"

    method = sys.argv[1] if len(sys.argv) > 1 else "RAB"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "MNIST"
    save_checkpoint = sys.argv[3] if len(sys.argv) > 3 else "N"
    if save_checkpoint == "N":
        from RunGurobi import GurobiBorder, GurobiFlip
        torch.set_default_dtype(torch.float64)

    if dataset_name == "Food101":
        initEpoch = 400
        G_epoch = 200
        # optimize = "SGD"

    if method == "RAB":
        n_samples_gurobi = -1
        if dataset_name == "EMNIST":
            n_samples_gurobi = 5000
    elif method == "RAF":
        n_samples_gurobi = 1000
        misclassification_count = 10

    train_dataset, test_dataset = GetDataset(dataset_name)

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_size = len(train_dataset)
    val_size = len(test_dataset)
    total_size = train_size + val_size
    total_run = 5

    for i in range(1, total_run + 1):
        model_t, model_g = GetModel(dataset_name, device=device)
    
        start_experiment = True if i == 1 else False

        rng = np.random.default_rng(seed=i*42)
        all_indices = rng.permutation(total_size)

        new_train_indices = all_indices[:train_size]
        new_val_indices = all_indices[train_size:]

        train_subset = Subset(full_dataset, new_train_indices)
        val_subset = Subset(full_dataset, new_val_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        learningRate = 0.01
        
        if os.path.exists(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth") == False:
            TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="Train", run_id=i, start_experiment=start_experiment)
            # try:
            TM.run()
            # except Exception as e:
            #     print(f"Error during training: {e}")
            #     total_run += 1
            #     continue
        if save_checkpoint == "Y":
            continue
        if os.path.exists(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint_GE_{method}.pth"):
            print(f"Checkpoint for run {i} already exists. Skipping Gurobi edit.")
            continue
        TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="GurobiEdit", run_id=i)

        if device.type == 'cuda':
            checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth")
        else:
            checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth", map_location=torch.device('cpu'))
        TM_after_g.model.load_state_dict(checkpoint['model_state_dict'])
        TM_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        TM_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded model for run {i} from checkpoint.")
        TM_after_g.save_fc_inputs("Train")
        TM_after_g.save_fc_inputs("Val")
        print(f"Saved FC inputs for run {i}.")
        if method == "RAB":
            Gurobi_output = GurobiBorder(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi)
        elif method == "RAF":
            Gurobi_output = GurobiFlip(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi, misclassification_count=misclassification_count)
        if Gurobi_output is None:
            print("Gurobi did not find a solution.")
            if total_run < 10:
                total_run += 1
            continue
        W2_new, b2_new = Gurobi_output
        TM_after_g.delete_fc_inputs()
        new_W = torch.tensor(W2_new).to(model_g.classifier.weight.device)
        new_b = torch.tensor(b2_new).to(model_g.classifier.bias.device)
        with torch.no_grad():
            TM_after_g.model.classifier.weight.copy_(new_W)
            TM_after_g.model.classifier.bias.copy_(new_b)
        train_loss, train_acc = TM_after_g.evaluate("Train")
        val_loss, val_acc = TM_after_g.evaluate("Val")

        with open(TM_after_g.log_file, "a") as f:
            f.write(f"{i},Gurobi_Complete_Eval_Train,-1,{train_loss},{train_acc}\n")
            f.write(f"{i},Gurobi_Complete_Eval_Val,-1,{val_loss},{val_acc}\n")
        try:
            TM_after_g.run()
        except Exception as e:
            print(f"Error during training: {e}")
            total_run += 1
            continue

