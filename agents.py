import torch
from scipy.linalg import lstsq

from neural_controller import NeuralController
from discrete_controller import DiscreteController




class Agent:
    def __init__(self, A, B, T, gamma, sigma, batch_size, optimality=None, n_gradient=100, net=None, rows=None, columns=None):
        self.A = A
        self.B = B
        self.net = net

        self.gamma = gamma
        self.sigma = sigma

        self.T = T
        self.d = A.shape[0]
        self.batch_size = batch_size
        self.n_gradient = n_gradient

        self.rows = rows if rows is not None else torch.ones(self.d, dtype=torch.bool)
        self.columns = columns if columns is not None else torch.ones(self.d, dtype=torch.bool)
        self.partial = rows is not None or columns is not None
        
        self.x_data = torch.zeros(1, self.d)
        self.y_data = torch.zeros(1, self.d)

        self.A_bar = A[2:, 2:]

        self.controller = DiscreteController(
            A,
            B,
            T,
            gamma=gamma,
            sigma=sigma,
            X_data=self.x_data,
            optimality=optimality,
            columns=self.columns
            )

        self.estimations = []
        self.optimality = optimality

        self.x = torch.zeros(self.d)

    
    def plan(self, A_hat, T):
        args = {
            'A': A_hat,
            'B': self.B,
            'X_data': self.x_data[:, self.columns],
            'T': T,
            'gamma': self.gamma,
            'sigma': self.sigma,
            'optimality': self.optimality,
            'columns': self.columns,
            'x': self.x
        }
        controller = DiscreteController
        if self.net is not None :
            args['net'] = self.net
            controller = NeuralController
        else:
            self.controller = controller(**args)

        # n_gradient = self.epoch_index * self.n_gradient
        self.controller.plan(self.n_gradient, self.batch_size)

    
    def play(self):
        with torch.no_grad():
            self.batch = torch.zeros(1, self.d)
            X, U = self.controller.play_control(self.batch, self.A)
            # print(f'played mean energy {torch.sum(U**2) / self.T}')
            energy_constraint = (torch.sum(U**2) / self.T <= (self.gamma**2)*1.1 )
            assert energy_constraint, f'energy constraint not met : mean energy {torch.sum(U**2) / self.T}'
        return X, U

    def play_random(self):
        self.batch = torch.zeros(1, self.d)
        self.controller.T = self.T
        with torch.no_grad():
            X, U = self.controller.play_random(
                self.batch, self.A)
        return X, U

    def update(self, X, U):
        if self.partial:
            return self.update_partial(X, U)
        Y = X[1:, :] - U@(self.B.T)
        self.x_data = torch.cat((self.x_data, X[:-1, :]), dim=0)
        self.y_data = torch.cat((self.y_data, Y), dim=0)
        solution, _, _, _ = lstsq(self.x_data, self.y_data)
        estimation = solution.T
        self.A_hat = torch.tensor(estimation)
        self.estimations.append(estimation.copy().reshape((1, self.d, self.d)))

        self.x = X[-1, :]

    def update_partial(self, X, U):
        
        Y = X[1:, :] - U@(self.B.T)
        self.x_data = torch.cat((self.x_data, X[:-1, :]), dim=0)
        self.y_data = torch.cat((self.y_data, Y[:, :]), dim=0)
        partial_planning = self.columns.sum() < self.d
        X_bar = self.x_data[:, 2:]
        X_tilde = self.x_data[:, :2]
        Y_tilde = self.y_data[:, 2:]
        solution, _, _, _ = lstsq(X_tilde, Y_tilde - X_bar@self.A_bar.T)
        estimation = solution.T
        self.A_tilde_hat = torch.tensor(estimation)
        self.A_hat = self.A.clone()
        self.A_hat[2:, :2] = self.A_tilde_hat
        print(self.A_hat - self.A)
        self.estimations.append(self.A_hat.numpy().copy().reshape((1, self.d, self.d)))


    def identify(self, n_steps):
        self.initialize()
        self.epoch_index = 1
        for epoch_index in range(n_steps):
            print(f'epoch {epoch_index}')
            self.T *= 2
            self.timestep()
            self.epoch_index += 1
        return self.estimations

    def identify_partial(self, n_steps):
        self.initialize()
        self.epoch_index = 1
        for epoch_index in range(n_steps):
            print(f'epoch {epoch_index}')
            self.T *= 2
            self.timestep()
            self.epoch_index += 1
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

