import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os


class BoundedControl(nn.Module):
    def __init__(self, net, gamma):
        self.gamma = gamma
        super().__init__()
        self.net = net

    def forward(self, x):
        output = self.net(x)
        return self.gamma * output / torch.norm(output, dim=1).unsqueeze(1)
