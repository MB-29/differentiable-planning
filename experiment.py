import numpy as np
from agents import *

class Experiment:

    def __init__(self, A, n_gradient, T0, sigma, gamma):

        self.A = A
        self.sigma = sigma
        self.gamma = gamma
        self.T0 = T0
        self.d = A.shape[0]
    
    def run(self, agent_map, n_gradient, n_epochs, n_samples):

        agent_outputs = {}

        for agent_name, agent_contructor in agent_map.items():
            method = agent_name.split(' ')[0]
            control = BoundedControl(net, gamma)
            agent = agent_contructor(
                self.A,
                control,
                self.T0,
                self.d,
                gamma=self.gamma,
                sigma=self.sigma,
                n_gradient=self.n_gradient,
                method=method)
            estimations = agent.identify(n_epochs, n_samples)
            residual = self.A.numpy() - estimations
            error_values = np.linalg.norm(residual, axis=(2, 3)).T
            if agent_name == 'active':
                gamma_sq_values = agent.gamma_sq_values

            agent_outputs[agent_name] = error_values


