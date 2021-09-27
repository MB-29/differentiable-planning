import numpy as np
from tqdm import tqdm
import torch

from agents import *



class Experiment:

    def __init__(self, A, d, T0, sigma, gamma, net=None):

        self.A = A
        self.d = d
        self.sigma = sigma
        self.gamma = gamma
        self.T0 = T0
        self.net = net

    def run(self, agent_map, n_gradient, n_epochs, n_samples):

        estimations = {}

        for agent_name, agent_contructor in agent_map.items():
            print(f'agent {agent_name}')
            agent_estimations = np.zeros((n_samples, n_epochs+1, self.d, self.d))
            for sample_index in tqdm(range(n_samples)):
                method = agent_name.split(' ')[0]
                A = self.get_A()
                agent = agent_contructor(
                    A,
                    self.T0,
                    self.d,
                    gamma=self.gamma,
                    sigma=self.sigma,
                    n_gradient=n_gradient,
                    method=method,
                    net=self.net
                    )
                sample_estimations = np.array(agent.identify(n_epochs)).squeeze()
                if self.A is None:
                    sample_estimations -= A.numpy()
                agent_estimations[sample_index, :, :, :] = sample_estimations
            estimations[agent_name] = agent_estimations
        return estimations
        
    def get_A(self):
        if self.A is not None:
            return self.A
        M = torch.randn(self.d, self.d)
        eigenvals = torch.linalg.eigvals(M)
        rho = torch.abs(eigenvals).max()
        return M / rho