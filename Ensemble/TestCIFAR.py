import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Networks.ResNet import ResNet18_CIFAR


BATCH_SIZE = 128
EPOCHS = 200
LR = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,transform=train_transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,transform=test_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


model = ResNet18_CIFAR(num_classes=10).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)


def train_one_epoch(epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for inputs, targets in tqdm(trainloader, desc=f"Train {epoch}"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    loss_avg = running_loss / len(trainloader)
    return loss_avg, acc


@torch.no_grad()
def test(epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    for inputs, targets in tqdm(testloader, desc=f"Test  {epoch}"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    loss_avg = running_loss / len(testloader)
    return loss_avg, acc

if __name__ == "__main__":
    best_test_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(epoch)
        test_loss, test_acc = test(epoch)

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_resnet_cifar10.pt")

    print(f"\nBest Test Accuracy: {best_test_acc:.2f}%")
