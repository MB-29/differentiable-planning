import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle
import torch.multiprocessing as mp

from agents import Active, Oracle, Random
from experiment import Experiment





# parameters

T0 = 10
n_samples = 5
n_epochs = 3
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


def identify_(agent_, optimality):
    agent = agent_(
        A,
        T0,
        d,
        gamma=gamma,
        sigma=sigma,
        n_gradient=n_gradient,
        optimality=optimality,
        net=net
    )
    estimation = np.array(agent.identify(n_epochs)).squeeze()
    residual = estimation - A.numpy()
    return residual

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

# experiment = Experiment(A, d, T0, sigma, gamma, net=net)
# t1 = time.time()
# # residuals = experiment.run(agents, n_gradient, n_epochs, n_samples)
# t2 = time.time()
# residuals = experiment.run_parallel(agents, n_gradient, n_epochs, n_samples)
# t3 = time.time()
# print(f'{t2-t1} {t3-t2}')
# # plot results




residuals = {}
n_processes = mp.cpu_count()



if __name__ == '__main__':
    print(f'Running on {n_processes} processes')

    for agent_name, agent_ in agents.items():
        print(f'agent {agent_name}')
        pool = mp.Pool(n_processes)
        optimality = agent_name.split(' ')[0]
        agent_residuals = np.zeros((n_samples, n_epochs+1, d, d))

        sample_results = [pool.apply_async(identify_, args=(agent_, optimality, )) for _ in range(n_samples)]
        for sample_index, result in enumerate(sample_results):
            agent_residuals[sample_index] = result.get()
        residuals[agent_name] = agent_residuals


    with open('output.pkl', 'wb') as f:
        pickle.dump(agent_residuals, f)

    for agent_name, agent_residuals in residuals.items():
        error_values = np.linalg.norm(agent_residuals, axis=(2, 3))
        mean = np.mean(error_values, axis=0)
        std = np.sqrt(np.var(error_values, axis=0) / n_samples)
        plt.errorbar(np.arange(n_epochs+1), mean, yerr=3 *std, label=agent_name, alpha=0.7)
    plt.yscale('log')
    plt.legend()
    plt.show()
