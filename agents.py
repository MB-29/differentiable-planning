import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os
from retry import retry

from controller import NeuralController
from discrete_controller import DiscreteController

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def estimate(X, U):
    Y = X[:, 1:, :] - U[:, :-1, :]
    A_hat = torch.linalg.lstsq(X[:, :-1, :], Y).solution.permute(0, 2, 1)
    return A_hat

class Agent:
    def __init__(self, A, control, T, d, gamma, method, sigma=1, batch_size=20, n_gradient=100):
        self.A = A
        self.controller = DiscreteController(A, control, d, T, gamma=gamma, sigma=sigma, method=method)


        self.gamma = gamma
        self.sigma = sigma

        self.control = control  
        self.T = T
        self.d = d
        self.batch_size = 16
        self.n_gradient = n_gradient
        self.dataset = torch.zeros((self.batch_size*n_gradient, d))

        self.x_data = []
        self.y_data = []

        self.estimations = []
        self.method = method
        architecture = method.split('-')[1]
        controllers = {
            'AD': DiscreteController,
            'random': DiscreteController,
            'adjoint': DiscreteController,
            'neural': NeuralController
            }
        self.controller_constructor = controllers[architecture]

    
    def plan(self, A_hat, T):
        # self.reset_weights()
        self.controller = self.controller_constructor(
            A_hat,
            self.control,
            self.d,
            T,
            method=self.method,
            gamma=self.gamma,
            sigma=self.sigma
            )

        train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)
        # max_epochs  =  max(200, self.n_epochs * self.T)
        trainer = pl.Trainer(
            max_epochs=1,
            checkpoint_callback=False
        )
        
        trainer.fit(self.controller, train_dataloader)

    
    def collect(self, T, n_samples=1):
        batch = torch.zeros(n_samples, self.d)
        X, U = self.controller(batch)
        Y = X[:, 1:, :] - U[:, :-1, :]
        for t in range(1, T-1):
            self.x_data.append(X[:, t].detach().numpy())
            self.y_data.append(Y[:, t].detach().numpy())
        return X, U

    
    def play(self, n_samples):
        # self.X = torch.Tensor(self.x_data)
        # self.Y = torch.Tensor(self.y_data)
        # solution = torch.linalg.lstsq(self.X, self.Y).solution.permute(0, 2, 1)
        # self.A_hat = solution.mean(dim=0)
        with torch.no_grad():
            self.batch = torch.zeros(n_samples, self.d)
            X, U = self.controller.play_control(self.batch, self.A)
        print(f'played control of energy {torch.norm(U[0, :, :])**2 / self.T}')
        return X, U

    def update(self, X, U):
        estimations = estimate(X, U)
        self.A_hat = estimations[0].detach()
        self.gamma_sq = (torch.sum(U**2, dim=(1,2)) / self.T).mean()
        self.estimations.append(estimations.detach().clone().numpy())

    def reset_weights(self):
        self.controller = self.controller_constructor(
            self.A_hat,
            self.control,
            self.d,
            self.T,
            method=self.method,
            gamma=self.gamma,
            sigma=self.sigma
            )
        for layer in self.control.net:
            if not hasattr(layer, 'reset_parameters'):
                return
            layer.reset_parameters()

    def identify(self, n_steps, n_samples):
        self.initialize(n_samples)
        self.gamma_sq_values = np.zeros(n_steps)
        for step_index in range(n_steps):
            self.T *= 2
            self.timestep(n_samples)
            self.gamma_sq_values[step_index] = self.gamma_sq
        return self.estimations

    def play_random(self, n_samples):
        self.batch = torch.zeros(n_samples, self.d)
        X, U = self.controller.play_random(self.batch, self.A, gamma=self.gamma)
        return X, U
    
    def initialize(self, n_samples):
        X, U = self.play_random(n_samples)
        estimations = estimate(X, U).detach()
        self.A_hat = estimations[0]
        self.estimations.append(estimations.detach().clone().numpy())


class Random(Agent):

    def timestep(self, n_samples):
        self.reset_weights()
        # self.collect(self.T)
        X, U = self.play(n_samples)
        self.update(X, U)
    
    def play(self, n_samples):
        return self.play_random(n_samples)

class Oracle(Agent):

    def timestep(self, n_samples):
        self.plan(self.A, self.T)
        # self.collect(self.T)
        X, U = self.play(n_samples)
        self.update(X, U)


class Active(Agent):

    def timestep(self, n_samples):
        self.plan(self.A_hat.detach().clone(), self.T)
        # self.collect(self.T)
        X, U = self.play(n_samples)
        self.update(X, U)

