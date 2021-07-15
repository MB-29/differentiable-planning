from mpl_toolkits.mplot3d import Axes3D
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class Controller(pl.LightningModule):
    def __init__(self, A, control, d, T, gamma, sigma, criterion='A-optimality'):
        super().__init__()
        self.control = control
        self.T = T
        self.A = A
        self.d = d

        self.gamma = gamma
        self.sigma = sigma

        loss_map = {
            'A-optimality': self.A_loss,
            'G-optimality': self.G_loss
        }
        self.loss_function = loss_map[criterion]


    def forward(self, x):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T, self.d)
        U = torch.zeros(batch_size, self.T, self.d)
        for t in range(self.T-1):
            time = torch.full((batch_size, 1), t)
            time_position = torch.cat((x, time), dim=1)
            control = self.control(time_position)
            x = (self.A @ x.T).T + control + self.sigma * torch.randn_like(x)
            U[:, t, :] = control
            X[:, t+1, :] = x
        return X, U

    def play(self, x, A):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T, self.d)
        U = torch.zeros(batch_size, self.T, self.d)
        for t in range(self.T-1):
            time = torch.full((batch_size, 1), t)
            time_position = torch.cat((x, time), dim=1)
            # control = self.control(time_position)
            control = self.control(time_position)
            x = (A @ x.T).T + control + self.sigma * torch.randn_like(x)
            U[:, t, :] = control
            X[:, t+1, :] = x
        return X, U

    def play_random(self, x, A, gamma):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T, self.d)
        U = torch.zeros(batch_size, self.T, self.d)
        for t in range(self.T-1):
            control = gamma*torch.randn_like(x) / self.d
            x = (A @ x.T).T + control + self.sigma * torch.randn_like(x)
            U[:, t, :] = control
            X[:, t+1, :] = x
        return X, U
    
    def trajectory_control(self, X):
        batch_size = X.shape[0]
        time_values = torch.linspace(0, self.T-1, self.T).unsqueeze(1).expand(batch_size, self.T, 1)
        time_position = torch.cat((X, time_values), dim=2)
        return self.control(time_position)
    
    def A_loss(self, S):
        return self.T * (1/S**2).sum(dim=1).mean()

    def G_loss(self, S):
        return - S[:, -1].mean() / self.T

    def training_step(self, batch, batch_idx):
        X, control_values = self.forward(batch)
        batch_size = X.shape[0]
        # X_ = torch.ones(batch_size, self.T, 2*self.d)
        # X_[:, :, :self.d] = X
        S = torch.linalg.svdvals(X)
        # S = torch.pca_lowrank(X)
        # S = torch.linalg.svdvals(X)
        # regularization = self.T * self.alpha * torch.sqrt(torch.mean(control_values[:, :, :]**2))
        # energy = torch.sum(control_values[:, 1:, :]**2, dim=(1,2)) / self.T
        # penalty = - torch.log(self.gamma**2 - energy).mean()
        # loss =  - torch.min(S, dim=1).values.mean() / self.T
        # loss =  self.T * (1/S**2).sum(dim=1).mean()
        # loss =  - S[:, -1].mean() / self.T 
        loss = self.loss_function(S)

        # loss += penalty
        # self.log("loss", loss)
        # trajectory = X_[0]
        # self.display(trajectory.detach())
        return loss
    
    # def validation_step(self, batch, batch_index):
    #     X, U = self.forward(batch)
    #     estimations = self.estimate(X, U)
    #     loss = torch.mean((estimations - self.A)**2)
    #     self.log("validation_loss", loss)
    
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.control.parameters(), lr=0.005)
    




