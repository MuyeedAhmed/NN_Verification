import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Subset
import random
import numpy as np

from Utils.CNNetworks import ResNet18, NIN_MNIST, NIN_EMNIST, VGG, CNN_USPS, VGG_office31

def GetModel(dataset_name, num_classes=10, device=None, extra_conv_layers=0):
    if dataset_name == "MNIST":
        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "CIFAR10":
        model_t = ResNet18(num_classes=num_classes).to(device)
        model_g = ResNet18(num_classes=num_classes).to(device)

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


def GetDataset(dataset_name, root_dir='./data', device=None, classCount=None):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
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


