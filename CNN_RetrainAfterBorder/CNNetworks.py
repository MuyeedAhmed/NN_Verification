import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import gurobipy as gp
from gurobipy import GRB
import numpy as np


class NIN_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN_CIFAR10, self).__init__()
        def nin_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            )
        self.features = nn.Sequential(
            nin_block(3, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nin_block(256, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nin_block(192, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nin_block(128, 64, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


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
        self.fc_hidden = nn.Linear(32*14*14, 16)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class NIN_FashionMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN_FashionMNIST, self).__init__()
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
        self.fc_hidden = nn.Linear(32*14*14, 16)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class NIN_KMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN_KMNIST, self).__init__()
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
        self.fc_hidden = nn.Linear(32*14*14, 16)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class NIN_EMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN_EMNIST, self).__init__()
        def nin_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            )
        self.features = nn.Sequential(
            nin_block(1, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nin_block(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(64*14*14, 32)
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


class NIN_SVHN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN_SVHN, self).__init__()
        def nin_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            )

        self.features = nn.Sequential(
            nin_block(3, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2, padding=1),
            nin_block(192, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            nin_block(192, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            nin_block(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(64*5*5, 64)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = self.flatten(x)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x