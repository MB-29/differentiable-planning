import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from matplotlib import rc, rcParams
from torchdyn.models.utils import DepthCat
from tqdm import tqdm 
import logging 

from agents import *
from controls import BoundedControl

# rc('font', size=30)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])


logging.getLogger("pytorch_lightning").setLevel(0)

T0 = 5
n_epochs = 10
gamma = 1
sigma = 0.5
n_gradient = 400

A = torch.randn(3, 3)
A = torch.tensor([

    [0.9, 1, 0, 0],
    [0, 0.9, 1, 0],
    [0, 0, 0.9, 1],
    [0, 0, 0, 0.9],
])
# A = torch.tensor([
#     [0.9, 1],
#     [0, 0.9]
# ])
# A = torch.zeros(2, 2)

d = A.shape[0]

net = nn.Sequential(
    nn.Linear(d+1, 16),
    nn.Tanh(),
    nn.Linear(16, d)
)

agent_map = {
    # 'discrete-active': Active,
    '-random': Random,
    # 'D-adjoint oracle': Oracle,
    # 'D-AD oracle': Oracle,
    # 'D-neural oracle': Oracle,
    # 'D-neural active': Active,
    # 'D-AD active': Active,
    # 'A-AD oracle': Oracle,
    # 'E-AD oracle': Oracle
}
color_map = {
    'active': 'darkblue',
    'random': 'darkred',
    'oracle': 'darkgreen',
    'adjoint': 'purple'
}

gamma_sq_values = np.zeros(n_epochs)

agent_outputs = {}
for agent_name, agent_contructor in agent_map.items():
    print(f'agent {agent_name}')
    method = agent_name.split(' ')[0]
    control = BoundedControl(net, gamma)
    agent = agent_contructor(
        A,
        control,
        T0,
        d,
        gamma=gamma,
        sigma=sigma,
        n_gradient=n_gradient,
        method=method)
    estimations = agent.identify(n_epochs)
    residual = A.numpy() - estimations
    error_values = np.linalg.norm(residual, axis=(2, 3)).T
    if agent_name == 'active':
        gamma_sq_values = agent.gamma_sq_values

    agent_outputs[agent_name] = error_values


for agent_name, error_values in agent_outputs.items():
    color = color_map.get(agent_name, 'black')
    mean = np.mean(error_values, axis=0)
    # std = np.sqrt(np.var(error_values, axis=0) / n_samples)
    plt.plot(np.arange(n_epochs+1), mean[:], label=agent_name, alpha=0.7)

plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('estimation error')
plt.title(r'A Jordan block of dimension 4')

plt.legend()
plt.show()
