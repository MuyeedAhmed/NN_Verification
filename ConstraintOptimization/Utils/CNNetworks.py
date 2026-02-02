import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchvision import models

import torch.nn.functional as F

import numpy as np

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet_OriginalHead(nn.Module):
    def __init__(self, num_classes=10, block=BasicBlock, num_blocks=(2,2,2,2)):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256, num_classes)

        self.fc_hidden = nn.Identity()
        self.classifier = self.fc

    def _make_layer(self, block, planes, nblocks, stride):
        strides = [stride] + [1]*(nblocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return out

    def forward(self, x, extract_fc_input=False):
        feats = self._features(x)
        if extract_fc_input:
            return feats.clone().detach(), None
        logits = self.fc(feats)
        return logits

def ResNet18(num_classes=10):
    return ResNet_OriginalHead(num_classes=num_classes, num_blocks=(2,2,2,2))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18_Small(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 3, block=BasicBlock, num_blocks=(2,2,2,2),
                 widths=(64, 128, 256, 256)):
        super().__init__()
        self.in_planes = widths[0]

        self.conv1 = nn.Conv2d(in_ch, widths[0], 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(widths[0])

        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        feat_dim = widths[3] * block.expansion
        self.fc = nn.Linear(feat_dim, num_classes)

        self.fc_hidden = nn.Identity()
        self.classifier = self.fc

    def _make_layer(self, block, planes, nblocks, stride):
        strides = [stride] + [1]*(nblocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return out

    def forward(self, x, extract_fc_input: bool = False):
        feats = self._features(x)
        if extract_fc_input:
            return feats.clone().detach(), None
        return self.fc(feats)


class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.shortcut = nn.Identity() if self.equal_in_out else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.equal_in_out else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WRNNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1,
                drop_rate
            ))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 3, depth: int = 28, widen_factor: int = 10, drop_rate: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor

        n_channels = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(in_ch, n_channels[0], 3, stride=1, padding=1, bias=False)
        self.block1 = WRNNetworkBlock(n, n_channels[0], n_channels[1], WRNBasicBlock, stride=1, drop_rate=drop_rate)
        self.block2 = WRNNetworkBlock(n, n_channels[1], n_channels[2], WRNBasicBlock, stride=2, drop_rate=drop_rate)
        self.block3 = WRNNetworkBlock(n, n_channels[2], n_channels[3], WRNBasicBlock, stride=2, drop_rate=drop_rate)

        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        feat_dim = n_channels[3]
        self.fc = nn.Linear(feat_dim, num_classes)

        self.fc_hidden = nn.Identity()
        self.classifier = self.fc

    def _features(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    def forward(self, x, extract_fc_input: bool = False):
        feats = self._features(x)
        if extract_fc_input:
            return feats.clone().detach(), None
        return self.fc(feats)


class ResNet50_BottleneckHead(nn.Module):
    def __init__(self, num_classes: int, bottleneck_dim: int = 256, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        backbone_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        self.fc_hidden = nn.Sequential(
            nn.Linear(backbone_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def _features(self, x):
        x = self.backbone(x)
        x = self.fc_hidden(x)
        return x

    def forward(self, x, extract_fc_input: bool = False):
        feats = self._features(x)
        if extract_fc_input:
            return feats.clone().detach(), None
        return self.classifier(feats)


def Net_SVHN(num_classes: int = 10):
    return WideResNet(num_classes=num_classes, in_ch=3, depth=28, widen_factor=10, drop_rate=0.0)

def Net_EMNIST(num_classes: int):
    return ResNet18_Small(num_classes=num_classes, in_ch=1)

def Net_KMNIST(num_classes: int = 10):
    return ResNet18_Small(num_classes=num_classes, in_ch=1)

def Net_FashionMNIST(num_classes: int = 10):
    return ResNet18_Small(num_classes=num_classes, in_ch=1)

def Net_USPS(num_classes: int = 10):
    return ResNet18_Small(num_classes=num_classes, in_ch=1)

def Net_Office31(pretrained: bool = True):
    return ResNet50_BottleneckHead(num_classes=31, pretrained=pretrained)

def Net_Food10(pretrained: bool = True):
    return ResNet50_BottleneckHead(num_classes=10, pretrained=pretrained)

def Net_Food101(pretrained: bool = True):
    return ResNet50_BottleneckHead(num_classes=101, pretrained=pretrained)

def Net_Caltech101(num_classes: int = 101, pretrained: bool = True, bottleneck_dim: int = 256):
    return ResNet50_BottleneckHead(num_classes=num_classes, bottleneck_dim=bottleneck_dim, pretrained=pretrained)


class NIN_MNIST(nn.Module):
    def __init__(self, num_classes=10, output_layer_size=16):
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
        self.fc_hidden = nn.Linear(32*14*14, output_layer_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_layer_size, num_classes)

    def forward(self, x, extract_fc_input=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        h = self.fc_hidden(x)
        h = self.relu(h)
        
        if extract_fc_input:
            return h.clone().detach(), None

        logits = self.classifier(h)
        return logits

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
        h = self.fc_hidden(x)
        h = self.relu(h)

        if extract_fc_input:
            return h.clone().detach(), None

        logits = self.classifier(h)
        return logits

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

        h = self.fc_hidden(x)
        h = self.relu(h)

        if extract_fc_input:
            return h.clone().detach(), None

        logits = self.classifier(h)
        return logits


class VGG(nn.Module):
    def __init__(self, num_classes=10, output_layer_size=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(256, output_layer_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_layer_size, num_classes)

    def _classifier_input(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc_hidden(x)
        x = self.relu(x)
        return x

    def forward(self, x, extract_fc_input=False):
        h = self._classifier_input(x)

        if extract_fc_input:
            return h.clone().detach(), None

        logits = self.classifier(h)
        return logits


class VGG_office31(nn.Module):
    def __init__(self, num_classes=31):
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
        
        h = self.fc_hidden(x)
        h = self.relu(h)

        if extract_fc_input:
            return h.clone().detach(), None

        logits = self.classifier(h)
        return logits
