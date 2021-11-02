import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from agents import Active, Oracle, Random
from experiment import Experiment

if __name__ == '__main__':
    # parameters

    T0 = 10
    n_samples = 100
    n_epochs = 5
    gamma = 1
    sigma = 0.1
    n_gradient = 20

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
        # 'random': Random,
        # 'A active': Active,
        'D active': Active,
        # 'E active': Active,
        # 'T active': Active,

        #'A oracle': Oracle,
        #'D oracle': Oracle,
        # 'E oracle': Oracle,
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
    t1 = time.time()
    residuals = experiment.run(agents, n_gradient, n_epochs, n_samples)
    t2 = time.time()
    # residuals = experiment.run_parallel(agents, n_gradient, n_epochs, n_samples)
    # t3 = time.time()
    print(f'Experiment run in {t2-t1} seconds for {n_samples} samples')
    # plot results

    for agent_name, agent_residuals in residuals.items():
        error_values = np.linalg.norm(agent_residuals, axis=(2, 3))
        mean = np.mean(error_values, axis=0)
        std = np.sqrt(np.var(error_values, axis=0) / n_samples)
        plt.errorbar(np.arange(n_epochs+1), mean, yerr=3 *std, label=agent_name, alpha=0.7)
    # plt.yscale('log')
    plt.legend()
    plt.show()
