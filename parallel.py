import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle
import multiprocessing as mp

from agents import Active, Oracle, Random
from experiment import Experiment





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
    'random': Random,
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






if __name__ == '__main__':
    t_start = time.time()
    
    residuals = {}
    n_processes = mp.cpu_count()
    print(f'Running on {n_processes} processes')


    for agent_name, agent_ in agents.items():
        print(f'agent {agent_name}')
        pool = mp.Pool(n_processes)
        optimality = agent_name.split(' ')[0]
        agent_residuals = np.zeros((n_samples, n_epochs+1, d, d))

        # ctx = mp.get_context('spawn')
        # with ctx.Pool(n_processes) as pool:
        sample_results = [pool.apply_async(identify_, args=(agent_, optimality, )) for _ in range(n_samples)]
        for sample_index, result in enumerate(sample_results):
            agent_residuals[sample_index] = result.get()
        residuals[agent_name] = agent_residuals

        print(f'simulations completed')

    t_end = time.time()
    print(f'Experiment run in {t_end-t_start} seconds for {n_samples} samples')


    output_name = f'{n_samples}-samples_{n_gradient}-gradient_{n_epochs}-epochs'
    
    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(agent_residuals, f)

