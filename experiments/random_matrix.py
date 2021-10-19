import numpy as np
import matplotlib.pyplot as plt
import logging 

from experiment import Experiment
from agents import Active, Oracle, Random

# parameters

T0 = 100
n_samples = 10
n_epochs = 5
gamma = 1
sigma = 0.1
n_gradient = 200

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

A = None 
d = 4
experiment = Experiment(A, d, T0, sigma, gamma)
estimations = experiment.run(agents, n_gradient, n_epochs, n_samples)


for agent_name, agent_estimations in estimations.items():
    residual = agent_estimations
    error_values = np.linalg.norm(residual, axis=(2, 3))
    mean = np.mean(error_values, axis=0)
    std = np.sqrt(np.var(error_values, axis=0) / n_samples)
    plt.errorbar(np.arange(n_epochs+1), mean, yerr=3 *std, label=agent_name, alpha=0.7)

plt.yscale('log')
plt.legend()
plt.show()
