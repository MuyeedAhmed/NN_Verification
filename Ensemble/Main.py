import torch
import csv
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader, Subset
from tqdm import tqdm
import os
import sys
import time
import random
import numpy as np
from TrainModel import TrainModel


from CNNetworks import ResNet18_CIFAR, NIN_MNIST, NIN_CIFAR10, NIN_SVHN, NIN_EMNIST, NIN, VGG, CNN_USPS, Food101Net, VGG_office31, VGG_var_layers


def GetModel(dataset_name, num_classes=10, device=None, output_layer_size=16, extra_conv_layers=0):
    if dataset_name == "MNIST":
        model_t = NIN_MNIST(num_classes=10, output_layer_size=output_layer_size).to(device)
        model_g = NIN_MNIST(num_classes=10, output_layer_size=output_layer_size).to(device)
    # elif dataset_name == "CIFAR10":
    #     if extra_conv_layers > 0:
    #         model_t = VGG_var_layers(num_classes=10, output_layer_size=output_layer_size, extra_conv_layers=extra_conv_layers).to(device)
    #         model_g = VGG_var_layers(num_classes=10, output_layer_size=output_layer_size, extra_conv_layers=extra_conv_layers).to(device)
    #     else:
    #         model_t = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
    #         model_g = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
    elif dataset_name == "CIFAR10":
        model_t = ResNet18_CIFAR(num_classes=num_classes).to(device)
        model_g = ResNet18_CIFAR(num_classes=num_classes).to(device)

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
    elif dataset_name == "PathMNIST":
        model_t = VGG(num_classes=9).to(device)
        model_g = VGG(num_classes=9).to(device)
    elif dataset_name == "Food101":
            model_t = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
            model_g = VGG(num_classes=num_classes, output_layer_size=output_layer_size).to(device)
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
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        all_classes = train_dataset.classes
        random.seed(42)
        if classCount is not None:
            selected_classes = random.sample(all_classes, classCount)
        else:
            selected_classes = random.sample(all_classes, 10)
        def fast_filter(dataset, selected_classes):
            class_names = dataset.classes
            label_names = [class_names[label] for label in dataset.targets]
            selected_indices = [i for i, name in enumerate(label_names) if name in selected_classes]
            return Subset(dataset, selected_indices)

        class_map = {cls_name: i for i, cls_name in enumerate(selected_classes)}
        train_filtered = fast_filter(train_dataset, selected_classes)
        test_filtered = fast_filter(test_dataset, selected_classes)

        train_dataset = RelabelSubset(train_filtered, class_map, train_dataset)
        test_dataset = RelabelSubset(test_filtered, class_map, test_dataset)

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

class WrapOneHotEncoding(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label = label.argmax().item()
        return image, label

    def __len__(self):
        return len(self.dataset)

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



@torch.no_grad()
def ensemble_test_accuracy(models, test_loader, device):
    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        probs_sum = None
        for m in models:
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

        avg_probs = probs_sum / len(models)
        pred = avg_probs.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / total


def load_models(checkpoint_paths, dataset_name, device, ols):
    models = []

    for path in checkpoint_paths:
        model, _ = GetModel(dataset_name, device=device, output_layer_size=ols)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models



@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    import torch.nn.functional as F
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total

if __name__ == "__main__":
    os.makedirs("Stats/RAB_CrossVal_All", exist_ok=True)
    os.makedirs("Stats/RAF_CrossVal_All", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    initEpoch = 300
    G_epoch = 0
    
    misclassification_count = 1
    top_k = 10
    total_candidates = 20

    method = sys.argv[1] if len(sys.argv) > 1 else "RAB"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "MNIST"
    save_checkpoint = sys.argv[3] if len(sys.argv) > 3 else "N"
    if save_checkpoint == "N":
        from RunGurobi import MILP

    n_samples_gurobi = 1000
        
    
    if dataset_name == "CIFAR10":
        BatchSize = 128
        optimize = "SGD"
        learningRate = 0.1
        scheduler_type = "MultiStepLR"
    else:
        BatchSize = 64
        optimize = "Adam"
        learningRate = 0.01
        scheduler_type = "CosineAnnealingLR"
    
    
    train_dataset, test_dataset = GetDataset(dataset_name)

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_size = int(len(train_dataset) * 0.8)
    val_size = int(len(train_dataset) * 0.2)
    total_size = train_size + val_size

    i = 2
    model_t, model_g = GetModel(dataset_name, device=device)

    rng = np.random.default_rng(seed=i*42)
    all_indices = rng.permutation(total_size)

    new_train_indices = all_indices[:train_size]
    new_val_indices = all_indices[train_size:]

    train_subset = Subset(train_dataset, new_train_indices)
    val_subset = Subset(train_dataset, new_val_indices)

    train_loader = DataLoader(train_subset, batch_size=BatchSize, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BatchSize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)
    

    if os.path.exists(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth") == False:
        TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="Train", run_id=i, start_experiment=True)
        TM.run()
    
    if save_checkpoint == "Y":
        sys.exit()

    results = []

    TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=BatchSize, learning_rate=learningRate, optimizer_type=optimize, scheduler_type=scheduler_type, phase="GurobiEdit", run_id=i)

    if device.type == 'cuda':
        checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth")
    else:
        checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run{i}_full_checkpoint.pth", map_location=torch.device('cpu'))
    
    TM_after_g.model.load_state_dict(checkpoint['model_state_dict'])
    TM_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    TM_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Loaded model for run {i} from checkpoint.")

    TM_after_g.save_fc_inputs("Train")
    TM_after_g.save_fc_inputs("Val")

    print(f"Saved FC inputs for run {i}.")
          
    for candidate in range(total_candidates):
        time0 = time.time()

        if candidate % 4 == 1:
            milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, candidate=0)
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Correct")
        elif candidate % 4 == 2:
            milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, candidate=0)
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Incorrect")
        elif candidate % 4 == 3:
            milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=n_samples_gurobi, tol=1e-5, misclassification_count=misclassification_count, candidate=0)
            Gurobi_output = milp_instance.Optimize(Method="MisCls_Any")
        else:
            milp_instance = MILP(dataset_name, TM_after_g.log_file, run_id=i, n=-1, tol=1e-5, candidate=0)
            Gurobi_output = milp_instance.Optimize(Method="LowerConf")

        time1 = time.time()
        
        if Gurobi_output is None:
            print("Gurobi did not find a solution.")
            continue

        W2_new, b2_new = Gurobi_output
        # TM_after_g.delete_fc_inputs()
        new_W = torch.tensor(W2_new).to(model_g.classifier.weight.device)
        new_b = torch.tensor(b2_new).to(model_g.classifier.bias.device)
        with torch.no_grad():
            TM_after_g.model.classifier.weight.copy_(new_W)
            TM_after_g.model.classifier.bias.copy_(new_b)
        checkpoint_dir = f"./checkpoints/{dataset_name}/Run{i}_checkpoint_{candidate}.pth"
        torch.save({
            'epoch': TM_after_g.num_epochs,
            'model_state_dict': TM_after_g.model.state_dict(),
            'optimizer_state_dict': TM_after_g.optimizer.state_dict(),
            'scheduler_state_dict': TM_after_g.scheduler.state_dict()
        }, checkpoint_dir)
        
        train_loss, train_acc = TM_after_g.evaluate("Train")
        val_loss, val_acc = TM_after_g.evaluate("Val")
        test_loss, test_acc = evaluate_loader(TM_after_g.model, test_loader, device)

        with open(TM_after_g.log_file, "a") as f:
            f.write(f"{i},{candidate},Gurobi_Complete_Eval_Train,-1,{train_loss},{train_acc}\n")
            f.write(f"{i},{candidate},Gurobi_Complete_Eval_Val,-1,{val_loss},{val_acc}\n")
            f.write(f"{i},{candidate},Gurobi_Complete_Eval_Test,-1,{test_loss},{test_acc}\n")

        results.append({
            "Candidate": candidate,
            "Checkpoint": checkpoint_dir,
            "Train_loss": float(train_loss),
            "Train_acc": float(train_acc),
            "Val_loss": float(val_loss),
            "Val_acc": float(val_acc),
            "Test_loss": float(test_loss),
            "Test_acc": float(test_acc),
            "Solve_Time": float(time1 - time0),
        })
        print(f"[Run {i} cand {candidate}] val_acc={val_acc:.4f} test_acc={test_acc:.4f} time={time1 - time0:.1f}s")
    
    TM_after_g.delete_fc_inputs()
    

    csv_path = "Stats/Summary.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Candidate","Checkpoint","Train_loss","Train_acc","Val_loss","Val_acc","Test_loss","Test_acc","Solve_Time",])

        if write_header:
            writer.writeheader()

        for row in results:
            writer.writerow(row)


    results_sorted = sorted(
        results,
        key=lambda r: r["Val_acc"],
        reverse=True
    )

    top_k_paths = [r["Checkpoint"] for r in results_sorted[:top_k]]
    # for r in top5_paths: error
    #     print(r["Candidate"], r["Val_acc"], r["Checkpoint"])


    models = load_models(top_k_paths, dataset_name, device, ols)

    ensemble_acc = ensemble_test_accuracy(
        models=models,
        test_loader=test_loader,
        device=device
    )

    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")