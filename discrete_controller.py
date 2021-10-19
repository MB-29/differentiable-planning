import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import pytorch_lightning as pl

from criteria import criteria

class DiscreteController(pl.LightningModule):
    def __init__(self, A, d, T, gamma, sigma, optimality=''):
        super().__init__()
        self.T = T
        self.A = A
        self.d = d

        self.gamma = gamma
        self.sigma = sigma

        self.U = torch.randn(self.T, self.d, requires_grad=True)

        self.criterion = criteria.get(optimality)


    def forward(self, x):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.play_dynamics(x, self.A, U, 0), U

    def play_dynamics(self, x, A, U, sigma):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T+1, self.d)
        for t in range(self.T):
            u =  U[t, :] 
            x = (A @ x.T).T + u 
            if sigma > 0:
                x += self.sigma * torch.randn_like(x)
            X[:, t+1, :] = x
        return X

    def play_control(self, x, A):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.play_dynamics(x, A, U, self.sigma), U

    def play_random(self, x, A, gamma):
        U = self.gamma * torch.randn(self.T, self.d) / np.sqrt(self.d)
        
        return self.play_dynamics(x, A, U, self.sigma), U
    
    
    def plan(self, n_steps, batch_size=1, learning_rate=0.1):
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        for step_index in range(n_steps):
            x = torch.zeros(batch_size, self.d)
            X, U = self.forward(x)
            S = torch.linalg.svdvals(X)
            loss = self.criterion(S, self.T).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return 
    


