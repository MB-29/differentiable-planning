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

from experiment import Experiment
from agents import *

# rc('font', size=30)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])


logging.getLogger("pytorch_lightning").setLevel(0)

T0 = 100
n_samples = 5
n_epochs = 3
gamma = 1
sigma = 0.1
n_gradient = 250

A = torch.tensor([

    [0.9, 1, 0, 0],
    [0, 0.9, 1, 0],
    [0, 0, 0.9, 1],
    [0, 0, 0, 0.9],
])


agent_map = {
    '-random': Random,
    # 'D-adjoint oracle': Oracle,
    # 'D-neural oracle': Oracle,
    # 'D-neural active': Active,
    # 'D-AD active': Active,
    'E-AD active': Active,

    # 'T-AD oracle': Oracle,
    'D-AD oracle': Oracle,
    'A-AD oracle': Oracle,
    'E-AD oracle': Oracle,
}

experiment = Experiment(A, T0, sigma, gamma)
estimations = experiment.run(agent_map, n_gradient, n_epochs, n_samples)

color_map = {
    'active': 'darkblue',
    'random': 'darkred',
    'oracle': 'darkgreen',
    'adjoint': 'purple'
}

for agent_name, agent_estimations in estimations.items():
    residual = A.numpy() - agent_estimations
    error_values = np.linalg.norm(residual, axis=(2, 3))
    color = color_map.get(agent_name, 'black')
    mean = np.mean(error_values, axis=0)
    std = np.sqrt(np.var(error_values, axis=0) / n_samples)
    plt.errorbar(np.arange(n_epochs+1), mean, yerr=3 *std, label=agent_name, alpha=0.7)

plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('estimation error')
plt.title(r'A Jordan block of dimension 4')

plt.legend()
plt.show()
