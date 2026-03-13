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


class TrainModel:
    def __init__(self, method, dataset_name, model, train_loader, val_loader, device, num_epochs=200, resume_epochs=100, batch_size=64, learning_rate=0.01, optimizer_type='SGD', scheduler_type='CosineAnnealingLR', phase = "Train", run_id=0):
        self.method = method
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.resume_epochs = resume_epochs
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
        
        if self.method == "AWP":
            self.awp = AWP(self.model, self.optimizer, adv_lr=1.0, adv_eps=0.01)
        else:
            self.awp = None

        os.makedirs("NNRunLog", exist_ok=True)
        self.log_file = f"NNRunLog/{self.dataset_name}.csv"
        
        if os.path.exists(self.log_file) == False:
            with open(self.log_file, "w") as f:
                f.write("Run,Phase,Method,Epoch,Train_loss,Train_acc,Val_loss,Val_acc\n")

    def train(self, early_stopping_patience=10, min_delta=1e-5, warmup_epochs=0):
        loss = -1
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = -1
        acceptable_val_acc = 0.0
        best_state_dict = None

        for epoch in range(self.num_epochs+self.resume_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for i, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            # for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)
                loss.backward()

                if self.awp is not None:
                    self.awp.attack_backward(inputs, labels_for_loss, self.criterion)
                    self.awp.restore()

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
            
                        
            if epoch + 1 <= warmup_epochs:
                with open(self.log_file, "a") as f:
                    f.write(f"{self.run_id},{self.phase},{self.method},{epoch+1},{avg_train_loss},{train_accuracy},-,-\n")
                continue
            
            val_loss, val_acc = self.evaluate("Val")
            with open(self.log_file, "a") as f:
                f.write(f"{self.run_id},{self.phase},{self.method},{epoch+1},{avg_train_loss},{train_accuracy},{val_loss},{val_acc}\n")

            # if best_train_loss - avg_train_loss > min_delta:
            #     best_train_loss = avg_train_loss
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1
            # if epochs_no_improve >= early_stopping_patience:
            #     print(f"Early stopping triggered after {epoch+1} epochs based on training loss.")
            #     break

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

            if epoch == self.num_epochs:
                if self.phase == "Train":
                    self.save_model(loss, save_suffix="")
                    test_accuracy = self.test()
                self.phase = "ResumeTrain"

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Restored best model from epoch {best_epoch}.")

        if self.phase == "Train":
            self.save_model(loss, save_suffix="")
        elif self.phase == "GurobiEdit":
            self.save_model(loss, save_suffix=f"_GE_{self.method}")
        elif self.phase == "ResumeTrain":
            self.save_model(loss, save_suffix="_Resume")
        
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels_for_loss = labels - 1 if self.dataset_name == "EMNIST" else labels
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)

                batch_size = inputs.size(0)
                total += batch_size
                total_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels_for_loss).sum().item()
                
        avg_loss = total_loss / total

        accuracy = 100. * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        with open(self.log_file, "a") as f:
            f.write(f"{self.run_id},{self.phase}_Test,{self.method},-1,{avg_loss},{accuracy},-,-\n")
        return accuracy

    def save_model(self, loss, save_suffix=""):
        if save_suffix == "" or save_suffix == "_Resume":
            checkpoint_dir = f"./checkpoints/{self.dataset_name}"
        else:
            checkpoint_dir = f"./checkpoints/{self.dataset_name}_CO"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # torch.save(self.model.fc_hidden.weight.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_fc_hidden_weight{save_suffix}.pt")
        # torch.save(self.model.fc_hidden.bias.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_fc_hidden_bias{save_suffix}.pt")
        torch.save(self.model.classifier.weight.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_classifier_weight{save_suffix}.pt")
        torch.save(self.model.classifier.bias.data.clone(), f"{checkpoint_dir}/Run{self.run_id}_classifier_bias{save_suffix}.pt")
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss.item()
        }, f"{checkpoint_dir}/Run{self.run_id}_full_checkpoint{save_suffix}.pth")

        # self.save_fc_inputs("Train", save_suffix=save_suffix)
        # self.save_fc_inputs("Val", save_suffix=save_suffix)

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
                # fc_input, _ = self.model(inputs, extract_fc_input=True)
                # logits = self.model.classifier(self.model.fc_hidden(fc_input))
                # # logits = self.model.classifier(torch.relu(self.model.fc_hidden(fc_input)))
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
            print(f"Files for not found for detele.")

    def run(self):
        start_time = time.time()
        if self.phase == "Train":
            self.train()
        elif self.phase == "GurobiEdit" or self.phase == "ResumeTrain":
            self.train(warmup_epochs=0)
        accuracy = self.test()
    
    def evaluate(self, dataset_type):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        if dataset_type == "Train":
            loader = self.train_loader
        elif dataset_type == "Val":
            loader = self.val_loader
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
        
        