import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.fc1 = nn.Linear(16 * 4 * 4, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=20, shuffle=False)

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

torch.save(model.state_dict(), "cnn_trained.pth")

fc1_w = model.fc1.weight.detach().numpy()
fc1_b = model.fc1.bias.detach().numpy()
fc2_w = model.fc2.weight.detach().numpy()
fc2_b = model.fc2.bias.detach().numpy()
fc3_w = model.fc3.weight.detach().numpy()
fc3_b = model.fc3.bias.detach().numpy()

np.savez("cnn_fc_weights.npz", fc1_w=fc1_w, fc1_b=fc1_b, fc2_w=fc2_w, fc2_b=fc2_b, fc3_w=fc3_w, fc3_b=fc3_b)

model.eval()
X_list = []
Z3_list = []

with torch.no_grad():
    for images, labels in testloader:
        x = model.pool(F.relu(model.conv1(images)))
        x = model.pool(F.relu(model.conv2(x)))
        x_flat = x.view(x.size(0), -1)
        z1 = model.fc1(x_flat)
        a1 = F.relu(z1)
        z2 = model.fc2(a1)
        a2 = F.relu(z2)
        z3 = model.fc3(a2)

        X_list.append(x_flat.numpy())
        Z3_list.append(z3.numpy())
        break

X = np.vstack(X_list)
Z3 = np.vstack(Z3_list)
np.savez("input_features_logits.npz", X=X, Z3=Z3)
