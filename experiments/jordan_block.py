import numpy as np
import matplotlib.pyplot as plt
import torch

from experiment import Experiment
from agents import Active, Oracle, Random

# parameters

T0 = 2
n_samples = 10
n_epochs = 2
gamma = 1
sigma = 0.1
n_gradient = 200

A = torch.tensor([

    [0.9, 1, 0, 0],
    [0, 0.9, 1, 0],
    [0, 0, 0.9, 1],
    [0, 0, 0, 0.9],
])
d = A.shape[0]

# run experiments with different agents
# choose agents by commenting in/out

agents = {
    'random': Random,
    # 'A active': Active,
    'D active': Active,
    # 'E active': Active,
    # 'T active': Active,

    #'A oracle': Oracle,
    #'D oracle': Oracle,
    'E oracle': Oracle,
    # 'T oracle': Oracle,
}

# comment out the neural net for a neural control (continuous time setting)

net = None
# net = control = nn.Sequential(
#     nn.Linear(d+1, 16),
#     nn.Tanh(),
#     nn.Linear(16, d)
# )

experiment = Experiment(A, d, T0, sigma, gamma, net=net)
estimations = experiment.run(agents, n_gradient, n_epochs, n_samples)

# plot results

for agent_name, agent_estimations in estimations.items():
    residual = A.numpy() - agent_estimations
    error_values = np.linalg.norm(residual, axis=(2, 3))
    mean = np.mean(error_values, axis=0)
    std = np.sqrt(np.var(error_values, axis=0) / n_samples)
    plt.errorbar(np.arange(n_epochs+1), mean, yerr=3 *std, label=agent_name, alpha=0.7)
plt.yscale('log')
plt.legend()
plt.show()
