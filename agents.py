import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
import os
from scipy.linalg import lstsq

from neural_controller import NeuralController
from discrete_controller import DiscreteController

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def estimate(X, U):
    Y = X[:, 1:, :] - U
    A_hat = torch.linalg.lstsq(X[:, :-1, :], Y).solution.permute(0, 2, 1)
    return A_hat

class Agent:
    def __init__(self, A, T, d, gamma, sigma, optimality=None, n_gradient=100, net=None):
        self.A = A
        self.controller = DiscreteController(A, d, T, gamma=gamma, sigma=sigma, optimality=optimality)
        self.net = net if net is not None else []
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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
        # self.reset_weights()
        if self.net != [] :
            self.controller = NeuralController(
                A_hat,
                self.d,
                T,
                self.net,
                optimality=self.optimality,
                gamma=self.gamma,
                sigma=self.sigma
                )
        else:
            self.controller = DiscreteController(
                A_hat,
                self.d,
                T,
                optimality=self.optimality,
                gamma=self.gamma,
                sigma=self.sigma
                )

        # train_dataloader = DataLoader(
        #     self.dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True
        #     )
        # # max_epochs  =  max(200, self.n_epochs * self.T)
        # trainer = pl.Trainer(
        #     max_epochs=1,
        #     checkpoint_callback=False,
        #     # progress_bar_refresh_rate=0
        # )
        
        # trainer.fit(self.controller, train_dataloader)
        
        self.controller.plan(self.n_gradient, self.batch_size)

    
    
    def play(self):
        # self.X = torch.Tensor(self.x_data)
        # self.Y = torch.Tensor(self.y_data)
        # solution = torch.linalg.lstsq(self.X, self.Y).solution.permute(0, 2, 1)
        # self.A_hat = solution.mean(dim=0)
        with torch.no_grad():
            self.batch = torch.zeros(1, self.d)
            X, U = self.controller.play_control(self.batch, self.A)
        print(f'played control of energy {torch.norm(U)**2 / self.T}')
        return X, U

    def play_random(self):
        self.batch = torch.zeros(1, self.d)
        self.controller.T = self.T
        X, U = self.controller.play_random(
            self.batch, self.A, gamma=self.gamma)
        print(f'T = {self.T}, played random control of energy {torch.norm(U)**2 / self.T}')
        return X, U

    def update(self, X, U):
        Y = X[1:, :] - U
        self.x_data = torch.cat((self.x_data, X[:-1, :]), dim=0)
        self.y_data = torch.cat((self.y_data, Y), dim=0)
        solution, _, _, _ = lstsq(self.x_data, self.y_data)
        estimation = solution.T
        # estimation = estimation.T
        # estimation = torch.linalg.lstsq(self.x_data, self.y_data).solution.T
        # estimation = torch.linalg.lstsq(X[:-1, :], Y).solution.T
        # estimations = estimate(X, U.unsqueeze(0))
        self.A_hat = torch.tensor(estimation)
        # self.gamma_sq = (torch.sum(U**2, dim=(1,2)) / self.T).mean()
        # self.estimations.append(self.A_hat.unsqueeze(0).clone().numpy())
        self.estimations.append(estimation.copy().reshape((1, self.d, self.d)))


    def identify(self, n_steps):
        self.initialize()
        # self.gamma_sq_values = np.zeros(n_steps)
        for step_index in range(n_steps):
            self.T *= 2
            self.timestep()
            # self.gamma_sq_values[step_index] = self.gamma_sq
        return self.estimations


    
    def initialize(self):
        X, U = self.play_random()
        self.update(X.squeeze(), U.squeeze())
        # estimations = estimate(X, U.unsqueeze(0)).detach()
        # self.A_hat = estimations[0]
        # self.estimations.append(estimations.detach().clone().numpy())
    


class Random(Agent):

    def timestep(self):
        X, U = self.play()
        self.update(X.squeeze(), U.squeeze())
    
    def play(self):
        return self.play_random()

class Oracle(Agent):

    def timestep(self):
        self.plan(self.A, self.T)
        # self.collect(self.T)
        X, U = self.play()
        self.update(X.squeeze(), U.squeeze())


class Active(Agent):

    def timestep(self):
        self.plan(self.A_hat.detach().clone().squeeze(), self.T)
        # self.collect(self.T)
        X, U = self.play()
        self.update(X.squeeze(), U.squeeze())

