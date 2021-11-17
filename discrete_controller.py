import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch

from utils import criteria, estimate, estimate_batch
from adjoint import Evaluation


class DiscreteController:
    def __init__(self, A, B, T, gamma, sigma,  X_data=None, optimality=''):
        super().__init__()
        self.T = T
        self.A = A
        self.B = B
        self.d, self.m = B.shape
        self.X_data = X_data if X_data is not None else torch.zeros(1, self.d)
        

        self.gamma = gamma
        self.sigma = sigma

        self.U = torch.randn(self.T, self.m, requires_grad=True)
        # self.U = torch.ones(self.T, self.m, requires_grad=True)

        self.criterion = criteria.get(optimality)


    def forward(self, x, stochastic=True):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.integration(x, self.A, U, stochastic), U

    def integration(self, x, A, U, stochastic):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T+1, self.d)
        for t in range(self.T):
            u =  U[t, :] 
            x = (A @ x.T).T + self.B@u 
            if stochastic:
                noise = self.sigma * torch.randn_like(x)
                x += noise
            X[:, t+1, :] = x
        # print(f'played mean energy {(U**2).sum()/self.T}')
        return X

    def play(self, x, A, U):
        # print(f'played mean energy {(U**2).sum() / self.T}')
        energy_constraint = (torch.sum(U**2) / self.T <= (self.gamma**2)*1.1)
        assert energy_constraint, f'energy constraint not met : mean energy {torch.sum(U**2) / self.T}'
        return self.integration(x, A, U, stochastic=True), U

    def play_control(self, x, A):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.play(x, A, U)

    def play_random(self, x, A):
        U = torch.randn(self.T, self.m)
        U_normalized = self.gamma * np.sqrt(self.T) * U / torch.norm(U)
        return self.play(x, A, U_normalized)
        
    
    def plan(self, n_steps, batch_size, stochastic=True, learning_rate=0.1, test=False):
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        loss_values = []
        error_values = []
        for step_index in range(n_steps):

            if test:
                # and int(100*step_index/n_steps)%10 == 0:
                test_loss, error = self.test(batch_size)
                # test_loss, error = self.test_batch(batch_size)
                
            
                loss_values.append(test_loss.item())
                error_values.append(error.item())

            x = torch.zeros(batch_size, self.d)
            X, U = self.forward(x, stochastic)
            X_data = self.X_data.unsqueeze(0).expand(batch_size, -1, -1)
            # print(f'{X_data.shape}, {X.shape}')
            X_total = torch.cat((X_data, X), dim=1)
            S = torch.linalg.svdvals(X_total[:, :-1, :])
            # print(S.min())
            loss = self.criterion(S, self.T)
            # print(f'loss {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.U.data = self.gamma *np.sqrt(self.T) * self.U / torch.norm(self.U)
            
        return loss_values, error_values

    def plan_adjoint(self, n_steps, batch_size, stochastic, learning_rate=0.1, test=False):
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        loss_values = []
        error_values = []
        for step_index in range(n_steps):

            if test:
                test_loss, error = self.test_batch(batch_size)
                # print(f'test loss {test_loss.item()}')
            
                loss_values.append(test_loss.item())
                error_values.append(error.item())

            U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
            loss = Evaluation.apply(self.A, self.B, self.U, self.T, self.sigma)
            # print(f'training loss {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.U.data = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)

            
        return loss_values, error_values
    

    def test(self, batch_size):
        with torch.no_grad():
            x = torch.zeros(1, self.d)
            X, U = self.play_control(x, self.A)
            # X, U = self.forward(x, False)
            S = torch.linalg.svdvals(X[:, :-1])
            test_loss = self.criterion(S, self.T)
            # M = X.permute(0, 2, 1) @ X.permute(0, 1, 2)
            # test_loss = - torch.log(torch.det(M)).mean()

            A_hat = estimate(X.squeeze(), U)
            error = torch.linalg.norm(A_hat - self.A)
            energy = torch.sum(U**2)/ self.T
            # print(f'X.shape {X.shape}, energy {energy}, A = {self.A}, A_hat = {A_hat}')
            # print(f'error {error}')
            return test_loss, error

    def test_batch(self, batch_size):
        with torch.no_grad():
            x = torch.zeros(batch_size, self.d)
            X, U = self.play_control(x, self.A)
            energy_constraint = (torch.sum(U**2) / self.T <= (self.gamma**2)*1.1)
            assert energy_constraint, f'energy constraint not met : mean energy {torch.sum(U**2) / self.T}'
            # X, U = self.forward(x, True)
            A_hat = estimate_batch(X, U.unsqueeze(0))
            error = torch.linalg.norm(A_hat - self.A, dim=(1,2)).mean()
            # print(f'test error {error}')

            S = torch.linalg.svdvals(X[:, :-1, :])
            # test_loss = self.criterion(S, self.T)
            test_loss = S[:, -1].mean()
            # M = X.permute(0, 2, 1) @ X.permute(0, 1, 2)
            # test_loss = - torch.log(torch.det(M)).mean()


            return test_loss, error

        
