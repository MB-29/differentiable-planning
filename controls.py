import torch.nn as nn
import torch


class BoundedControl(nn.Module):
    def __init__(self, net, gamma):
        self.gamma = gamma
        super().__init__()
        self.net = net

    def forward(self, x):
        output = self.net(x)
        return self.gamma * output / torch.norm(output, dim=1).unsqueeze(1)
