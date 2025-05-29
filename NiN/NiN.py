import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class NINNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(NINNetwork, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), 
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 10)   # 256 → 10
        self.fc2 = nn.Linear(10, 10)           # 10 → 10
        self.classifier = nn.Linear(10, num_classes)  # 10 → 10 (for MNIST)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x



name_dataset = "mnist"

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=15, shuffle=False)
print(len(trainloader))
print(len(testset))
model = NINNetwork()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training Started")
for epoch in range(5):
    model.train()
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

print("Training Finished")


# BLOCK 1
block1_conv0_w = model.block1[0].weight.detach().cpu().numpy()
block1_conv0_b = model.block1[0].bias.detach().cpu().numpy()
block1_conv2_w = model.block1[2].weight.detach().cpu().numpy()
block1_conv2_b = model.block1[2].bias.detach().cpu().numpy()
block1_conv4_w = model.block1[4].weight.detach().cpu().numpy()
block1_conv4_b = model.block1[4].bias.detach().cpu().numpy()

# BLOCK 2
block2_conv0_w = model.block2[0].weight.detach().cpu().numpy()
block2_conv0_b = model.block2[0].bias.detach().cpu().numpy()
block2_conv2_w = model.block2[2].weight.detach().cpu().numpy()
block2_conv2_b = model.block2[2].bias.detach().cpu().numpy()
block2_conv4_w = model.block2[4].weight.detach().cpu().numpy()
block2_conv4_b = model.block2[4].bias.detach().cpu().numpy()

fc1_w = model.fc1.weight.detach().cpu().numpy()
fc1_b = model.fc1.bias.detach().cpu().numpy()
fc2_w = model.fc2.weight.detach().cpu().numpy()
fc2_b = model.fc2.bias.detach().cpu().numpy()
# CLASSIFIER
classifier_w = model.classifier.weight.detach().cpu().numpy()
classifier_b = model.classifier.bias.detach().cpu().numpy()

# Save all
np.savez("nin_block_weights.npz",
    block1_conv0_w=block1_conv0_w, block1_conv0_b=block1_conv0_b,
    block1_conv2_w=block1_conv2_w, block1_conv2_b=block1_conv2_b,
    block1_conv4_w=block1_conv4_w, block1_conv4_b=block1_conv4_b,
    block2_conv0_w=block2_conv0_w, block2_conv0_b=block2_conv0_b,
    block2_conv2_w=block2_conv2_w, block2_conv2_b=block2_conv2_b,
    block2_conv4_w=block2_conv4_w, block2_conv4_b=block2_conv4_b,
    fc1_w=fc1_w, fc1_b=fc1_b,
    fc2_w=fc2_w, fc2_b=fc2_b,
    classifier_w=classifier_w, classifier_b=classifier_b
)


model.eval()
X_list = []
Z3_list = []

with torch.no_grad():
    for images, labels in testloader:
        x = model.block1(images)
        x = model.block2(x)
        x_flat = x.view(x.size(0), -1)  # (batch, 256)
        z1 = model.fc1(x_flat)
        a1 = F.relu(z1)
        z2 = model.fc2(a1)
        a2 = F.relu(z2)
        z3 = model.classifier(a2)

        X_list.append(x_flat.cpu().numpy())
        Z3_list.append(z3.cpu().numpy())
        break  # <-- REMOVE THIS to process all test samples

X = np.vstack(X_list)   # Remove axis=0 argument; np.vstack does not use it
Z3 = np.vstack(Z3_list)
np.savez("input_features_logits.npz", X=X, Z3=Z3)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

