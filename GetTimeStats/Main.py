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


from CNNetworks import NIN_MNIST, NIN_CIFAR10, NIN_SVHN, NIN_EMNIST, NIN, VGG, CNN_USPS, Food101Net, VGG_office31, VGG_var_layers


def GetModel(dataset_name, num_classes=10, device=None, output_layer_size=16, extra_conv_layers=0):
    if dataset_name == "MNIST":
        model_t = NIN_MNIST(num_classes=10, output_layer_size=output_layer_size).to(device)
        model_g = NIN_MNIST(num_classes=10, output_layer_size=output_layer_size).to(device)
    elif dataset_name == "CIFAR10":
        if extra_conv_layers > 0:
            model_t = VGG_var_layers(num_classes=10, output_layer_size=output_layer_size, extra_conv_layers=extra_conv_layers).to(device)
            model_g = VGG_var_layers(num_classes=10, output_layer_size=output_layer_size, extra_conv_layers=extra_conv_layers).to(device)
        else:
            model_t = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
            model_g = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
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
            model_t = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
            model_g = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
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

def GetDataset(dataset_name, root_dir='./data', device=None, classCount=None):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        all_classes = train_dataset.classes
        random.seed(42)
        if classCount is not None:
            selected_classes = random.sample(all_classes, classCount)
        else:
            selected_classes = random.sample(all_classes, 10)
        def fast_filter(dataset, selected_classes):
            class_names = dataset.classes
            label_names = [class_names[label] for label in dataset.targets]
            selected_indices = [i for i, name in enumerate(label_names) if name in selected_classes]
            return Subset(dataset, selected_indices)

        class_map = {cls_name: i for i, cls_name in enumerate(selected_classes)}
        train_filtered = fast_filter(train_dataset, selected_classes)
        test_filtered = fast_filter(test_dataset, selected_classes)

        train_dataset = RelabelSubset(train_filtered, class_map, train_dataset)
        test_dataset = RelabelSubset(test_filtered, class_map, test_dataset)

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
        if classCount is not None:
            selected_classes = random.sample(all_classes, classCount)
        else:
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
    initEpoch = 50
    G_epoch = 1
    optimize = "Adam"

    method = sys.argv[1] if len(sys.argv) > 1 else "RAB"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "MNIST"
    save_checkpoint = sys.argv[3] if len(sys.argv) > 3 else "N"
    if save_checkpoint == "N":
        from RunGurobi import GurobiBorder, GurobiFlip_Any, GurobiFlip_Correct
        
    # if dataset_name == "Food101":
    #     initEpoch = 400
    #     G_epoch = 200
    #     # optimize = "SGD"

    if method == "RAB":
        n_samples_gurobi = -1
    elif method == "RAF":
        n_samples_gurobi = 1000
        misclassification_count = 1

    train_dataset, test_dataset = GetDataset(dataset_name)

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_size = len(train_dataset)
    val_size = len(test_dataset)
    total_size = train_size + val_size

    test = "classes" # "nodes" or "samples" or "layers"

    if test == "nodes":
        ol_sizes = [16, 32, 64, 128, 256, 512, 1024]
        for l_size in ol_sizes:
            total_run = 5
            for i in range(1, total_run + 1):
                model_t, model_g = GetModel(dataset_name, device=device, output_layer_size=l_size)
            
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
                
                TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="Train", run_id=i, start_experiment=start_experiment)
                
                TM.run()
                
                if save_checkpoint == "Y":
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
                time0 = time.time()
                if method == "RAB":
                    Gurobi_output = GurobiBorder(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi)
                elif method == "RAF":
                    Gurobi_output = GurobiFlip_Any(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi, misclassification_count=misclassification_count)
                    # Gurobi_output = GurobiFlip_Correct(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi, misclassification_count=misclassification_count)
                time1 = time.time()
                
                
                if Gurobi_output is None:
                    print("Gurobi did not find a solution.")
                    if total_run < 10:
                        total_run += 1
                else:
                    with open("Stats/TimeStats.txt", "a") as f:
                        f.write(f"{dataset_name},{l_size},Run{i},{method},{time1 - time0}\n")
                
    elif test == "samples":
        total_run = 5
        for i in range(1, total_run + 1):
            if dataset_name == "MNIST":
                ols = 16
            elif dataset_name == "CIFAR10":
                ols = 128
            model_t, model_g = GetModel(dataset_name, device=device, output_layer_size=ols)
        
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
            
            TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="Train", run_id=i, start_experiment=start_experiment)
            
            TM.run()
            
            if save_checkpoint == "Y":
                continue
            
            TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="GurobiEdit", run_id=i)

            if device.type == 'cuda':
                checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth")
            else:
                checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth", map_location=torch.device('cpu'))
            
            TM_after_g.model.load_state_dict(checkpoint['model_state_dict'])
            TM_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            TM_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            TM_after_g.save_fc_inputs("Train")
            TM_after_g.save_fc_inputs("Val")

            # n_samples_gurobis = [100, 500, 1000, 5000, 10000]
            n_samples_gurobis = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
            for n_samples_gurobi in n_samples_gurobis:
                time0 = time.time()
                if method == "RAB":
                    Gurobi_output = GurobiBorder(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi)
                elif method == "RAF":
                    Gurobi_output = GurobiFlip_Any(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi, misclassification_count=misclassification_count)
                time1 = time.time()
                
                
                if Gurobi_output is None:
                    print("Gurobi did not find a solution.")
                    if total_run < 10:
                        total_run += 1
                else:
                    with open("Stats/TimeStats_SampleSize.txt", "a") as f:
                        f.write(f"{dataset_name},{n_samples_gurobi},Run{i},{method},{time1 - time0}\n")
    elif test == "layers":
        # extra_layers = [1, 2, 3, 4, 5]
        extra_layers = [6, 7, 8, 9]
        for el in extra_layers:
            total_run = 5
            for i in range(1, total_run + 1):
                if dataset_name == "MNIST":
                    ols = 16
                elif dataset_name == "CIFAR10":
                    ols = 128
                model_t, model_g = GetModel(dataset_name, device=device, output_layer_size=ols, extra_conv_layers=el)
            
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
                
                TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="Train", run_id=i, start_experiment=start_experiment)
                
                TM.run()
                
                if save_checkpoint == "Y":
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
                time0 = time.time()
                if method == "RAB":
                    Gurobi_output = GurobiBorder(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi)
                elif method == "RAF":
                    Gurobi_output = GurobiFlip_Any(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi, misclassification_count=misclassification_count)
                time1 = time.time()
                
                
                if Gurobi_output is None:
                    print("Gurobi did not find a solution.")
                    if total_run < 10:
                        total_run += 1
                else:
                    with open("Stats/TimeStats_ExtraLayers.txt", "a") as f:
                        f.write(f"{dataset_name},{el},Run{i},{method},{time1 - time0}\n")
    elif test == "classes":
        classCounts = [2, 4, 6, 8, 10]
        if dataset_name == "MNIST":
            ols = 16
        elif dataset_name == "CIFAR10" or dataset_name == "Food101":
            ols = 128
        total_run = 5
        for i in range(1, total_run + 1):
            for cc in classCounts:
                train_dataset, test_dataset = GetDataset(dataset_name, classCount=cc)

                full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
                train_size = len(train_dataset)
                val_size = len(test_dataset)
                total_size = train_size + val_size

                model_t, model_g = GetModel(dataset_name, device=device, output_layer_size=ols, num_classes=cc)
            
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
                if dataset_name == "Food101":
                    learningRate = 0.05
                TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="Train", run_id=i, start_experiment=start_experiment)
                
                TM.run()
                
                if save_checkpoint == "Y":
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
                time0 = time.time()
                if method == "RAB":
                    Gurobi_output = GurobiBorder(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi)
                elif method == "RAF":
                    Gurobi_output = GurobiFlip_Any(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi, misclassification_count=misclassification_count)
                time1 = time.time()
                
                
                if Gurobi_output is None:
                    print("Gurobi did not find a solution.")
                    # if total_run < 10:
                    #     total_run += 1
                else:
                    with open("Stats/TimeStats_Classes.txt", "a") as f:
                        f.write(f"{dataset_name},{cc},Run{i},{method},{time1 - time0}\n")
    else:
        total_run = 5
        for i in range(1, total_run + 1):
            if dataset_name == "MNIST":
                ols = 16
            elif dataset_name == "CIFAR10":
                ols = 128
            elif dataset_name == "Food101":
                ols = 256
            model_t, model_g = GetModel(dataset_name, device=device, output_layer_size=ols)
        
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
                
                TM.run()
            
            if save_checkpoint == "Y":
                continue
            
            TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="GurobiEdit", run_id=i)

            if device.type == 'cuda':
                checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth")
            else:
                checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth", map_location=torch.device('cpu'))
            
            TM_after_g.model.load_state_dict(checkpoint['model_state_dict'])
            TM_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            TM_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            TM_after_g.save_fc_inputs("Train")
            TM_after_g.save_fc_inputs("Val")

            # n_samples_gurobis = [100, 500, 1000, 5000, 10000]
            n_samples_gurobis = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000]
            for n_samples_gurobi in n_samples_gurobis:
                time0 = time.time()
                if method == "RAB":
                    Gurobi_output = GurobiBorder(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi)
                elif method == "RAF":
                    Gurobi_output = GurobiFlip_Any(dataset_name, TM_after_g.log_file, i, n=n_samples_gurobi, misclassification_count=misclassification_count)
                time1 = time.time()
                
                
                if Gurobi_output is None:
                    print("Gurobi did not find a solution.")
                    if total_run < 10:
                        total_run += 1
                else:
                    with open("Stats/TimeStats_SampleSize.txt", "a") as f:
                        f.write(f"{dataset_name},{n_samples_gurobi},Run{i},{method},{time1 - time0}\n")