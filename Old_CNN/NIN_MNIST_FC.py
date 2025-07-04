import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class CifarNIN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNIN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(192, 160, 1), nn.ReLU(),
            nn.Conv2d(160, 96, 1), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),

            nn.Conv2d(96, 192, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(192, 192, 1), nn.ReLU(),
            nn.Conv2d(192, 192, 1), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),

            nn.Conv2d(192, 192, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(192, 192, 1), nn.ReLU(),
            # nn.Conv2d(192, 10, 1), nn.ReLU()
            nn.Conv2d(192, 2, 1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # self.fc_hidden = nn.Linear(10 * 8 * 8, 256)
        # self.classifier = nn.Linear(256, num_classes)
        self.fc_hidden = nn.Linear(2 * 8 * 8, 12)
        self.classifier = nn.Linear(12, num_classes)


    def forward(self, x, return_hidden_input=False):
        x = self.features(x)
        x_flat = self.flatten(x)
        h = F.relu(self.fc_hidden(x_flat))
        out = self.classifier(h)
        if return_hidden_input:
            return out, x_flat.detach(), self.fc_hidden.weight.detach().clone()
        return out

def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader

def train_one_epoch(model, loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    loop = tqdm(enumerate(loader), total=len(loader), desc="Training", ncols=100)
    for batch_idx, (images, labels) in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"\n[Epoch Completed] Average Loss: {avg_loss:.4f}\n")


def save_fc_data(model, loader, device, out_dir="./CIFER_1E"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    all_X_fc, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, X_fc, _ = model(x, return_hidden_input=True)
            all_X_fc.append(X_fc.cpu())
            all_labels.append(y)
    torch.save(torch.cat(all_X_fc), f"{out_dir}/X_fc.pt")
    torch.save(torch.cat(all_labels), f"{out_dir}/labels.pt")
    torch.save({
        "fc_hidden_weight": model.fc_hidden.weight.detach().cpu(),
        "fc_hidden_bias": model.fc_hidden.bias.detach().cpu(),
        "classifier_weight": model.classifier.weight.detach().cpu(),
        "classifier_bias": model.classifier.bias.detach().cpu()
    }, f"{out_dir}/weights.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CifarNIN().to(device)
    loader = get_cifar10_loaders()
    train_one_epoch(model, loader, device)
    save_fc_data(model, loader, device)
