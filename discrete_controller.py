import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch

from utils import criteria, estimate, estimate_batch


class DiscreteController:
    def __init__(self, A, d, T, gamma, sigma, optimality=''):
        super().__init__()
        self.T = T
        self.A = A
        self.d = d

        self.gamma = gamma
        self.sigma = sigma

        self.U = torch.randn(self.T, self.d, requires_grad=True)

        self.criterion = criteria.get(optimality)


    def forward(self, x, stochastic):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.integration(x, self.A, U, stochastic), U

    def integration(self, x, A, U, stochastic):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T+1, self.d)
        for t in range(self.T):
            u =  U[t, :] 
            x = (A @ x.T).T + u 
            if stochastic:
                x += self.sigma * torch.randn_like(x)
            X[:, t+1, :] = x
        return X

    def play_control(self, x, A):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.integration(x, A, U, self.sigma), U

    def play_random(self, x, A, gamma):
        U = self.gamma * torch.randn(self.T, self.d) / np.sqrt(self.d)
        
        return self.integration(x, A, U, self.sigma), U
    
    def plan(self, n_steps, batch_size, stochastic=True, learning_rate=0.1, test=False):
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        loss_values = []
        error_values = []
        for step_index in range(n_steps):
            x = torch.zeros(batch_size, self.d)
            X, U = self.forward(x, stochastic)
            S = torch.linalg.svdvals(X[:, :-1, :])
            loss = self.criterion(S, self.T)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if test:
                test_loss, error = self.test(batch_size)
            
                loss_values.append(test_loss.item())
                error_values.append(error.item())
            
        return loss_values, error_values
    
    def test(self, batch_size):
        with torch.no_grad():
            x = torch.zeros(batch_size, self.d)
            X, U = self.play_control(x, self.A)
            S = torch.linalg.svdvals(X[:, :-1, :])
            test_loss = self.criterion(S, self.T)

            A_hat = estimate_batch(X, U.unsqueeze(0))
            error = torch.linalg.norm(A_hat - self.A, dim=(1,2)).mean()

            return test_loss, error

        
    


