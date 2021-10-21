import numpy as np
from tqdm import tqdm
import torch
import multiprocessing as mp

from agents import *


def identify_parallel(experiment, agent_, n_gradient, n_epochs, optimality):
    return experiment.identify(agent_, n_gradient, n_epochs, optimality)


class Experiment:

    def __init__(self, A, d, T0, sigma, gamma, net=None):

        self.A = A
        self.d = d
        self.sigma = sigma
        self.gamma = gamma
        self.T0 = T0
        self.net = net

    def run(self, agents, n_gradient, n_epochs, n_samples):

        residuals = {}

        for agent_name, agent_ in agents.items():
            print(f'agent {agent_name}')
            agent_residuals = np.zeros((n_samples, n_epochs+1, self.d, self.d))
            optimality = agent_name.split(' ')[0]
            for sample_index in tqdm(range(n_samples)):
                A = self.get_A()
                agent = agent_(
                    A,
                    self.T0,
                    self.d,
                    gamma=self.gamma,
                    sigma=self.sigma,
                    n_gradient=n_gradient,
                    optimality=optimality,
                    net=self.net
                    )
                sample_estimations = np.array(agent.identify(n_epochs)).squeeze()
                # if self.A is None:
                sample_residuals = sample_estimations - A.numpy()
                agent_residuals[sample_index, :, :, :] = sample_residuals
            residuals[agent_name] = agent_residuals
        return residuals
    
    def run_parallel(self, agents, n_gradient, n_epochs, n_samples):
        residuals = {}

        n_processes = 4
        print(f'Running on {n_processes} processes')

        agent_residuals = np.zeros((n_samples, n_epochs+1, self.d, self.d))
        for agent_name, agent_ in agents.items():
            print(f'agent {agent_name}')
            optimality = agent_name.split(' ')[0]
            args = (self, agent_, n_gradient, n_epochs, optimality, )

            ctx = mp.get_context('spawn')
            with ctx.Pool(n_processes) as pool:
                sample_results = [pool.apply_async(identify_parallel, args=args) for _ in range(n_samples)]
            for sample_index, result in enumerate(sample_results):
                agent_residuals[sample_index] = result.get()
            residuals[agent_name] = agent_residuals
        return residuals
    
    def identify(self, agent_, n_gradient, n_epochs, optimality):
        A = self.get_A()
        agent = agent_(
            A,
            self.T0,
            self.d,
            gamma=self.gamma,
            sigma=self.sigma,
            n_gradient=n_gradient,
            optimality=optimality,
            net=self.net
            )
        estimation = np.array(agent.identify(n_epochs)).squeeze()
        residual = estimation - A.numpy()
        return residual


        
    def get_A(self):
        if self.A is not None:
            return self.A
        M = torch.randn(self.d, self.d)
        eigenvals = torch.linalg.eigvals(M)
        rho = torch.abs(eigenvals).max()
        return M / rho
