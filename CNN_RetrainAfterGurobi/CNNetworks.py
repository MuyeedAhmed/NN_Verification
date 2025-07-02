import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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
            nin_block(128, 128, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)

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


class CNN_USPS(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_USPS, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(128 * 4 * 4, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.relu1(self.conv1(x))
        x = self.pool1(self.relu2(self.conv2(x)))
        x = self.pool2(self.relu3(self.conv3(x)))
        x = self.flatten(x)
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
            nin_block(1, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nin_block(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(64*14*14, 64)
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


class NIN_SVHN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN_SVHN, self).__init__()
        def nin_block(in_ch, out_ch, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=1), nn.ReLU()
            )

        self.features = nn.Sequential(
            nin_block(3, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nin_block(192, 160, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nin_block(160, 96, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(96, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)


    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = self.flatten(x)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(64 * 8 * 8, 64)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = torch.relu(x)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = self.flatten(x)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class Food101Net(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), 
            nn.Dropout2d(p=0.5),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = self.flatten(x)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class VGG_office31(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 16x16
            nn.Dropout2d(p=0.5),

            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 8x8
            nn.Dropout2d(p=0.5),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = self.flatten(x)
        if extract_fc_input:
            return x.clone().detach(), None
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x