import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(96, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x, extract_fc_input=False):
        for i in range(12):
            x = self.features[i](x)
        if extract_fc_input:
            fc_input = x.clone().detach()
        x = self.features[12](x)
        x = self.features[13](x)
        x = x.view(x.size(0), -1)
        if extract_fc_input:
            return x, fc_input
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

model = NIN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
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
print(f"Test Accuracy after initial training: {100. * correct / total:.2f}%")

os.makedirs("./checkpoints", exist_ok=True)
torch.save(model.features[12].weight.data.clone(), "./checkpoints/last_weight_original.pt")
torch.save(model.features[12].bias.data.clone(), "./checkpoints/last_bias_original.pt")

model.eval()
X_fc = []
Y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _, fc_input = model(inputs, extract_fc_input=True)
        X_fc.append(fc_input.cpu())
        Y_true.append(labels.cpu())

X_fc = torch.cat(X_fc, dim=0).view(len(test_dataset), -1)
torch.save(X_fc, "./checkpoints/fc_inputs.pt")
torch.save(torch.cat(Y_true, dim=0), "./checkpoints/fc_labels.pt")
