import torch

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
import copy
import numpy as np


class AWP:
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1.0, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data.copy_(torch.max(
                        torch.min(param.data, self.backup_eps[name][1]),
                        self.backup_eps[name][0]
                    ))

    def attack_backward(self, inputs, labels, criterion):
        self._save()
        self._attack_step()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = criterion(logits, labels)
        loss.backward()

    def restore(self):
        self._restore()


class SAM:
    def __init__(self, model, optimizer, rho=0.05):
        self.model = model
        self.optimizer = optimizer
        self.rho = rho
        self.backup = {}

    @torch.no_grad()
    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()

    @torch.no_grad()
    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    @torch.no_grad()
    def _attack_step(self):
        grads = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.view(-1))
        
        if not grads:
            return
            
        grad_norm = torch.norm(torch.cat(grads))
        scale = self.rho / (grad_norm + 1e-12)

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                e_w = param.grad * scale
                param.data.add_(e_w)

    def attack_backward(self, inputs, labels, criterion):
        self._save()
        self._attack_step()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = criterion(logits, labels)
        loss.backward()

    def restore(self):
        self._restore()


class RWP:
    def __init__(self, model, optimizer, adv_param="weight", eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.eps = eps
        self.backup = {}

    @torch.no_grad()
    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                self.backup[name] = param.data.clone()

    @torch.no_grad()
    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    @torch.no_grad()
    def _attack_step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                noise = torch.randn_like(param.data)
                norm_n = torch.norm(noise)
                norm_p = torch.norm(param.data)
                if norm_n != 0:
                    r_at = self.eps * noise / norm_n * norm_p
                    param.data.add_(r_at)

    def attack_backward(self, inputs, labels, criterion):
        self._save()
        self._attack_step()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = criterion(logits, labels)
        loss.backward()

    def restore(self):
        self._restore()


class TrainModel:
    def __init__(self, training_type, dataset_name, model, train_loader, val_loader, device, test_loader, num_epochs=200, batch_size=64, learning_rate=0.01, optimizer_type='SGD', scheduler_type='CosineAnnealingLR', phase = "Train", run_id=0):
        self.training_type = training_type
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.phase = phase
        if optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        elif optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        else:
            raise ValueError("Unsupported optimizer type. Use 'SGD', 'Adam', or 'AdamW'.")

        if scheduler_type == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        elif scheduler_type == "MultiStepLR":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(0.5*self.num_epochs), int(0.75*self.num_epochs)], gamma=0.1)
       
        self.criterion = nn.CrossEntropyLoss()

        if "AWP" in self.training_type:
            self.awp = AWP(self.model, self.optimizer, adv_lr=1.0, adv_eps=0.01)
        else:
            self.awp = None
        
        if "SAM" in self.training_type:
            self.sam = SAM(self.model, self.optimizer, rho=0.05)
        else:
            self.sam = None
            
        if "RWP" in self.training_type:
            self.rwp = RWP(self.model, self.optimizer, eps=0.01)
        else:
            self.rwp = None

        os.makedirs("NNRunLog", exist_ok=True)
        self.log_file = f"NNRunLog/{self.dataset_name}.csv"
        
        if os.path.exists(self.log_file) == False:
            with open(self.log_file, "w") as f:
                f.write("Run,Phase,TrainingType,Epoch,Train_loss,Train_acc,Val_loss,Val_acc,Test_loss,Test_acc\n")
                
    def train(self, early_stopping_patience=25, min_delta=1e-5, save_suffix="", co_dir=False, co_subdir=""):
        loss = -1
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = -1
        acceptable_val_acc = 0.0
        best_state_dict = None

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for i, (inputs, labels) in enumerate(tqdm(self.train_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)
                loss.backward()

                if self.awp is not None:
                    self.awp.attack_backward(inputs, labels_for_loss, self.criterion)
                    self.awp.restore()

                if self.sam is not None:
                    self.sam.attack_backward(inputs, labels_for_loss, self.criterion)
                    self.sam.restore()

                if self.rwp is not None:
                    self.rwp.attack_backward(inputs, labels_for_loss, self.criterion)
                    self.rwp.restore()

                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels_for_loss).sum().item()

            if self.scheduler is not None:
                self.scheduler.step()
            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100. * correct / total

            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')

            val_loss, val_acc = self.evaluate("Val")
            test_loss, test_acc = self.evaluate("Test")
            with open(self.log_file, "a") as f:
                f.write(f"{self.run_id},{self.phase},{self.training_type},{epoch+1},{avg_train_loss},{train_accuracy},{val_loss},{val_acc},{test_loss},{test_acc}\n")

            if early_stopping_patience is not None:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch + 1
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience and val_acc >= acceptable_val_acc:
                    print(f"Early stopping at epoch {epoch+1}. Best was epoch {best_epoch} (val_loss={best_val_loss:.4f}).")
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Restored best model from epoch {best_epoch}.")

        self.save_model(loss, save_suffix=save_suffix, co_dir=co_dir, co_subdir=co_subdir)

    def checkpoint_paths(self, save_suffix="", co_dir=False, co_subdir=""):
        if "AWP" in self.training_type:
            tt = "AWP"
        elif "SAM" in self.training_type:
            tt = "SAM"
        elif "RWP" in self.training_type:
            tt = "RWP"
        else:
            tt = "ERM"
        checkpoint_dir = f"./checkpoints_{tt}/{self.dataset_name}"
        if co_dir:
            checkpoint_dir += "_CO"
            if co_subdir:
                checkpoint_dir += f"/{co_subdir}"
        full_ckpt_path = f"{checkpoint_dir}/Run{self.run_id}_full_checkpoint{save_suffix}.pth"
        weight_path = f"{checkpoint_dir}/Run{self.run_id}_classifier_weight{save_suffix}.pt"
        bias_path = f"{checkpoint_dir}/Run{self.run_id}_classifier_bias{save_suffix}.pt"
        return checkpoint_dir, full_ckpt_path, weight_path, bias_path

    def save_model(self, loss, save_suffix="", co_dir=False, co_subdir=""):
        checkpoint_dir, full_ckpt_path, weight_path, bias_path = self.checkpoint_paths(save_suffix, co_dir, co_subdir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(self.model.classifier.weight.data.clone(), weight_path)
        torch.save(self.model.classifier.bias.data.clone(), bias_path)
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss.item() if hasattr(loss, "item") else loss
        }, full_ckpt_path)

    def load_model(self, save_suffix="", co_dir=False, co_subdir=""):
        _, full_ckpt_path, _, _ = self.checkpoint_paths(save_suffix, co_dir, co_subdir)
        map_location = None if self.device.type == 'cuda' else torch.device('cpu')
        checkpoint = torch.load(full_ckpt_path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def save_fc_inputs(self, dataset_type, save_suffix=""):
        checkpoint_dir_input = f"./checkpoints_inputs/{self.dataset_name}"
        os.makedirs(checkpoint_dir_input, exist_ok=True)
        self.model.eval()
        X_fc_input = []
        Y_true = []
        Y_pred = []
        if dataset_type == "Train":
            loader = self.train_loader
            save_suffix = "_train" + save_suffix
        elif dataset_type == "Val":
            loader = self.val_loader
            save_suffix = "_val" + save_suffix
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                fc_input, _ = self.model(inputs, extract_fc_input=True)
                logits = self.model.classifier(fc_input)
                
                preds = torch.argmax(logits, dim=1)
                X_fc_input.append(fc_input.cpu())
                Y_true.append(labels.cpu())
                Y_pred.append(preds.cpu())

        X_fc_input = torch.cat(X_fc_input, dim=0)
        Y_true = torch.cat(Y_true, dim=0)
        Y_pred = torch.cat(Y_pred, dim=0)

        torch.save(X_fc_input, f"{checkpoint_dir_input}/fc_inputs{save_suffix}.pt")
        torch.save(Y_true, f"{checkpoint_dir_input}/fc_labels{save_suffix}.pt")
        torch.save(Y_pred, f"{checkpoint_dir_input}/fc_preds{save_suffix}.pt")
    
    def delete_fc_inputs(self):
        checkpoint_dir_input = f"./checkpoints_inputs/{self.dataset_name}"
        
        try:
            os.remove(f"{checkpoint_dir_input}/fc_inputs_train.pt")
            os.remove(f"{checkpoint_dir_input}/fc_labels_train.pt")
            os.remove(f"{checkpoint_dir_input}/fc_preds_train.pt")
            os.remove(f"{checkpoint_dir_input}/fc_inputs_val.pt")
            os.remove(f"{checkpoint_dir_input}/fc_labels_val.pt")
            os.remove(f"{checkpoint_dir_input}/fc_preds_val.pt")
        except FileNotFoundError:
            print(f"Files for not found for deletion.")

    def run(self, early_stopping_patience=50, min_delta=1e-5, save_suffix="", co_dir=False, co_subdir=""):
        self.train(early_stopping_patience=early_stopping_patience, min_delta=min_delta, save_suffix=save_suffix, co_dir=co_dir, co_subdir=co_subdir)

    def evaluate(self, dataset_type):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        if dataset_type == "Train":
            loader = self.train_loader
        elif dataset_type == "Val":
            loader = self.val_loader
        elif dataset_type == "Test":
            loader = self.test_loader
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)
                batch_size = inputs.size(0)

                total_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels_for_loss).sum().item()
                total += batch_size

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy
        