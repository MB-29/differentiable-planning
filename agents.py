import pytorch_lightning as pl
import torch
import os
from scipy.linalg import lstsq

from neural_controller import NeuralController
from discrete_controller import DiscreteController



def estimate(X, U):
    Y = X[:, 1:, :] - U
    A_hat = torch.linalg.lstsq(X[:, :-1, :], Y).solution.permute(0, 2, 1)
    return A_hat

class Agent:
    def __init__(self, A, T, d, gamma, sigma, optimality=None, n_gradient=100, net=None):
        self.A = A
        self.controller = DiscreteController(A, d, T, gamma=gamma, sigma=sigma, optimality=optimality)
        self.net = net

        self.gamma = gamma
        self.sigma = sigma

        self.T = T
        self.d = d
        self.batch_size = 16
        self.n_gradient = n_gradient
        self.dataset = torch.zeros((self.batch_size*n_gradient, d))

        self.x_data = torch.zeros(1, self.d)
        self.y_data = torch.zeros(1, self.d)

        self.estimations = []
        self.optimality = optimality

    
    def plan(self, A_hat, T):
        args = {
            'A': A_hat,
            'd': self.d,
            'T': T,
            'gamma': self.gamma,
            'sigma': self.sigma,
            'optimality': self.optimality
        }
        controller = DiscreteController
        if self.net is not None :
            args['net'] = self.net
            controller = NeuralController
        else:
            self.controller = controller(**args)

        
        self.controller.plan(self.n_gradient, self.batch_size)

    
    
    def play(self):
        with torch.no_grad():
            self.batch = torch.zeros(1, self.d)
            X, U = self.controller.play_control(self.batch, self.A)
        return X, U

    def play_random(self):
        self.batch = torch.zeros(1, self.d)
        self.controller.T = self.T
        X, U = self.controller.play_random(
            self.batch, self.A, gamma=self.gamma)
        return X, U

    def update(self, X, U):
        Y = X[1:, :] - U
        self.x_data = torch.cat((self.x_data, X[:-1, :]), dim=0)
        self.y_data = torch.cat((self.y_data, Y), dim=0)
        solution, _, _, _ = lstsq(self.x_data, self.y_data)
        estimation = solution.T
        self.A_hat = torch.tensor(estimation)
        self.estimations.append(estimation.copy().reshape((1, self.d, self.d)))


    def identify(self, n_steps):
        self.initialize()
        for step_index in range(n_steps):
            self.T *= 2
            self.timestep()
        return self.estimations


    def initialize(self):
        X, U = self.play_random()
        self.update(X.squeeze(), U.squeeze())
    


class Random(Agent):

    def timestep(self):
        X, U = self.play()
        self.update(X.squeeze(), U.squeeze())
    
    def play(self):
        return self.play_random()

class Oracle(Agent):

    def timestep(self):
        self.plan(self.A, self.T)
        X, U = self.play()
        self.update(X.squeeze(), U.squeeze())


class Active(Agent):

    def timestep(self):
        self.plan(self.A_hat.detach().clone().squeeze(), self.T)
        X, U = self.play()
        self.update(X.squeeze(), U.squeeze())

