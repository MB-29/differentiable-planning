import numpy as np
from tqdm import tqdm
import torch.nn as nn

from agents import *
from controls import BoundedControl




class Experiment:

    def __init__(self, A, T0, sigma, gamma):

        self.A = A
        self.sigma = sigma
        self.gamma = gamma
        self.T0 = T0
        self.d = A.shape[0]
    
    def run(self, agent_map, n_gradient, n_epochs, n_samples):

        estimations = {}

        net = nn.Sequential(
            nn.Linear(self.d+1, 16),
            nn.Tanh(),
            nn.Linear(16, self.d)
        )
        for agent_name, agent_contructor in agent_map.items():
            print(f'agent {agent_name}')
            agent_estimations = np.zeros((n_samples, n_epochs+1, self.d, self.d))
            for sample_index in tqdm(range(n_samples)):
                method = agent_name.split(' ')[0]
                control = BoundedControl(net, self.gamma)
                agent = agent_contructor(
                    self.A,
                    control,
                    self.T0,
                    self.d,
                    gamma=self.gamma,
                    sigma=self.sigma,
                    n_gradient=n_gradient,
                    method=method
                    )
                sample_estimations = np.array(agent.identify(n_epochs)).squeeze()
                agent_estimations[sample_index, :, :, :] = sample_estimations
            estimations[agent_name] = agent_estimations
        return estimations

                # residual = self.A.numpy() - estimations
                # error_values = np.linalg.norm(residual, axis=(2, 3)).T
                # if agent_name == 'active':
                #     gamma_sq_values = agent.gamma_sq_values

                # agent_outputs[agent_name] = error_values
