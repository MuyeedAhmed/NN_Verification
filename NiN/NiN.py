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

        
        self.classifier = nn.Sequential(
            nn.Conv2d(16, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


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

# CLASSIFIER
classifier_conv_w = model.classifier[0].weight.detach().cpu().numpy()
classifier_conv_b = model.classifier[0].bias.detach().cpu().numpy()

# Save all
np.savez("nin_block_weights.npz",
    block1_conv0_w=block1_conv0_w, block1_conv0_b=block1_conv0_b,
    block1_conv2_w=block1_conv2_w, block1_conv2_b=block1_conv2_b,
    block1_conv4_w=block1_conv4_w, block1_conv4_b=block1_conv4_b,
    block2_conv0_w=block2_conv0_w, block2_conv0_b=block2_conv0_b,
    block2_conv2_w=block2_conv2_w, block2_conv2_b=block2_conv2_b,
    block2_conv4_w=block2_conv4_w, block2_conv4_b=block2_conv4_b,
    classifier_conv_w=classifier_conv_w, classifier_conv_b=classifier_conv_b
)

X_list = []
Z3_list = []

model.eval()  

with torch.no_grad():
    for images, labels in testloader:
        x = model.block1(images)      
        x = model.block2(x)           
        X_list.append(x.cpu().numpy())

        
        out = model.classifier(x)
        z3 = out.view(out.size(0), -1).cpu().numpy()
        Z3_list.append(z3)
        break

X = np.concatenate(X_list, axis=0)

Z3 = np.concatenate(Z3_list, axis=0)  
np.savez("input_features_logits.npz", X=X, Z3=Z3)