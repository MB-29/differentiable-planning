import torch

from controls import BoundedControl
from utils import criteria


class NeuralController:
    def __init__(self, A, d, T, net, gamma, sigma, optimality=None):
        super().__init__()
     
        self.control = BoundedControl(net, gamma)
        self.T = T
        self.A = A
        self.d = d

        self.gamma = gamma
        self.sigma = sigma


        self.U = torch.randn(self.T, self.d, requires_grad=True)

        self.optimality = optimality
        self.criterion = criteria.get(optimality)


    def forward(self, x):
        return self.play_dynamics(x, self.A, self.sigma)

    def play_control(self, x, A):
        return self.play_dynamics(x, A, self.sigma)

    def play_dynamics(self, x, A, sigma):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T+1, self.d)
        U = torch.zeros(self.T, self.d)
        for t in range(self.T):
            time = torch.full((batch_size, 1), t)
            position_time = torch.cat((x, time), dim=1)
            u = self.control(position_time)[0, :]
            x = (A @ x.T).T + u
            if sigma > 0:
                x += self.sigma * torch.randn_like(x)
            X[:, t+1, :] = x
            U[t, :] = u
        return X, U
    
    

    def plan(self, n_steps, batch_size=1, learning_rate=0.1):
        optimizer = torch.optim.Adam(self.control.parameters(), lr=learning_rate)
        for step_index in range(n_steps):
            x = torch.zeros(batch_size, self.d)
            X, U = self.forward(x)
            S = torch.linalg.svdvals(X)
            loss = self.criterion(S, self.T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return
    




