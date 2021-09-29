import numpy as np
import matplotlib.pyplot as plt
import torch
import logging 

from experiment import Experiment
from agents import *



logging.getLogger("pytorch_lightning").setLevel(0)

T0 = 100
n_samples = 1
n_epochs = 8
gamma = 1
sigma = 0.1
n_gradient = 250

A = torch.tensor([

    [0.9, 1, 0, 0],
    [0, 0.9, 1, 0],
    [0, 0, 0.9, 1],
    [0, 0, 0, 0.9],
])
d = A.shape[0]

# Choose agents by commenting in/out
agent_map = {
    '-random': Random,
    # 'D-adjoint oracle': Oracle,
    # 'D-neural oracle': Oracle,
    # 'E-neural oracle': Oracle,
    # 'E-neural active': Active,
    #'D-AD active': Active,
    'E-AD active': Active,

    # 'T-AD oracle': Oracle,
    #'D-AD oracle': Oracle,
    #'A-AD oracle': Oracle,
    'E-AD oracle': Oracle,
}

experiment = Experiment(A, d, T0, sigma, gamma)
estimations = experiment.run(agent_map, n_gradient, n_epochs, n_samples)


for agent_name, agent_estimations in estimations.items():
    residual = A.numpy() - agent_estimations
    error_values = np.linalg.norm(residual, axis=(2, 3))
    mean = np.mean(error_values, axis=0)
    std = np.sqrt(np.var(error_values, axis=0) / n_samples)
    plt.errorbar(np.arange(n_epochs+1), mean, yerr=3 *std, label=agent_name, alpha=0.7)

plt.show()