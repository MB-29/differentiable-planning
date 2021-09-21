from mpl_toolkits.mplot3d import Axes3D
import os

from torchdyn.models.utils import DepthCat

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchdyn.models import NeuralDE
from information import Information

from dynamics import Dynamics


class DiscreteController(pl.LightningModule):
    def __init__(self, A, control, d, T, gamma, sigma, method):
        super().__init__()
        self.control = control
        self.T = T
        self.A = A
        self.d = d

        self.gamma = gamma
        self.sigma = sigma

        self.U = torch.randn(self.T-1, self.d, requires_grad=True)

        training_steps = {
            'D-AD': self.training_step_D,
            'E-AD': self.training_step_E,
            'A-AD': self.training_step_A,
            'D-adjoint': self.training_step_adjoint
        }

        self.training_step = training_steps.get(method)

        print(f'controller with method {method}')

    def forward(self, x):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.play_dynamics(x, self.A, U, 0)

    def play_dynamics(self, x, A, control_values, sigma):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T, self.d)
        U = torch.zeros(batch_size, self.T, self.d)
        for t in range(self.T-1):
            u =  control_values[t, :] 
            x = (A @ x.T).T + u 
            if sigma > 0:
                x += self.sigma * torch.randn_like(x)
            X[:, t+1, :] = x
            U[:, t, :] = u
        # print(f'energy {torch.norm(U[0])}')
        return X, U

    def play_control(self, x, A):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        # print(f'playing control of energy {torch.norm(U)**2 / self.T}')
        return self.play_dynamics(x, A, U, self.sigma)

    def play_random(self, x, A, gamma):
        control_values = torch.randn_like(x) / np.sqrt(self.d)
        return self.play_dynamics(x, A, control_values, self.sigma)
        # batch_size = x.shape[0]
        # X = torch.zeros(batch_size, self.T, self.d)
        # U = torch.zeros(batch_size, self.T, self.d)
        # for t in range(self.T-1):
        #     control = gamma*
        #     x = (A @ x.T).T + control + self.sigma * torch.randn_like(x)
        #     U[:, t, :] = control
        #     X[:, t+1, :] = x
        # return X, U
    
    def trajectory_control(self, X):
        batch_size = X.shape[0]
        time_values = torch.linspace(0, self.T-1, self.T).unsqueeze(1).expand(batch_size, self.T, 1)
        position_time = torch.cat((X, time_values), dim=2)
        return self.control(position_time)
    
    def A_loss(self, S):
        return self.T * (1/S**2).sum(dim=1).mean()

    def G_loss(self, S):
        return - S[:, -1].mean() / self.T

    def training_step_A(self, batch, batch_idx):
        X, control_values = self.forward(batch)
        S = torch.linalg.svdvals(X)
        loss = - (1/self.T) * S.sum(dim=1).mean()
        return loss

    def training_step_E(self, batch, batch_idx):
        X, control_values = self.forward(batch)
        S = torch.linalg.svdvals(X)
        loss = - S[:, -1].mean()
        return loss
        
    def training_step_D(self, batch, batch_idx):
        X, control_values = self.forward(batch)
        batch_size = X.shape[0]
        design_matrix = X.permute(0, 2, 1) @ X
        loss =  -torch.log(torch.det(design_matrix)).mean()
        return loss

    def training_step_adjoint(self, batch, batch_idx):
        # print(self.U)
        # U = np.sqrt(self.T) * self.gamma * self.U / torch.norm(self.U)
        U = self.U
        # print(torch.norm(self.U)**2 / self.T)
        loss = Information().apply(U, self.A, self.T)
        return loss

    def display(self, trajectory, close=True):
        fig = plt.figure()
        # ax = Axes3D(fig)    
        ax = plt.gca()    
        ax.scatter(trajectory[:, 0], trajectory[:, 1], alpha=.7, marker='x')
        size = self.T * self.gamma
        ax.set_xlim((-size, size))
        ax.set_ylim((-size, size))
        # ax.set_zlim((-size, size))
        plt.pause(0.1)
        if close:
            plt.close()
    
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.control.parameters(), lr=0.005)
    def configure_optimizers(self):
        return torch.optim.Adam([self.U], lr=0.005)
    


