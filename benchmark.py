import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from matplotlib import rc, rcParams
from tqdm import tqdm 
import logging 

from agents import *
from controls import BoundedControl

# rc('font', size=30)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])


logging.getLogger("pytorch_lightning").setLevel(0)

T = 5
n_samples = 1000
n_steps = 5
gamma = 1
sigma = 0.1
n_epochs = 200

A = torch.randn(3, 3)

d = A.shape[0]

net = nn.Sequential(
    nn.Linear(d+1, 16),
    nn.Tanh(),
    nn.Linear(16, d)
)

agent_map = {
    'active': Active,
    'random': Random,
    'oracle': Oracle
}
color_map = {
    'active': 'darkblue',
    'random': 'darkred',
    'oracle': 'darkgreen'
}

gamma_sq_values = np.zeros(n_steps)

agent_outputs = {}
for agent_name, agent_contructor in agent_map.items():
    print(f'agent {agent_name}')
    control = BoundedControl(net, gamma)
    agent = agent_contructor(A, control, T, d, gamma=gamma, sigma=sigma, n_epochs=n_epochs)
    estimations = agent.explore(n_steps, n_samples)
    residual = A.numpy() - estimations
    error_values = np.linalg.norm(residual, axis=(2, 3)).T
    if agent_name == 'active':
        gamma_sq_values = agent.gamma_sq_values

    agent_outputs[agent_name] = error_values


oracle_mean = np.mean(agent_outputs['oracle'], axis=0)
for agent_name, error_values in agent_outputs.items():
    color = color_map[agent_name]
    mean = np.mean(error_values, axis=0)
    std = np.sqrt(np.var(error_values, axis=0) / n_samples)
    plt.errorbar(np.arange(n_steps+1), mean[:], yerr=3 *std[:], label=agent_name, alpha=0.5, color=color)

plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('estimation error')
plt.title(r'estimation of randomly generated A of dimension 3')

plt.legend()
plt.show()
