import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(64, 1)
        self.A_last_cache = None

    def forward(self, x, store_A_last=False):
        x = self.net(x)
        if store_A_last:
            self.A_last_cache = x.detach().clone()
        return self.final_layer(x)

    def manual_forward_final(self, A_last, W, b):
        return F.linear(A_last, W, b)