import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle
import multiprocessing as mp
import sys

from agents import Active, Oracle, Random

T0 = 100
n_samples = 100
n_epochs = 8
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
B = torch.eye(d)

def identify_(agent_, optimality):
    agent = agent_(
        A,
        B,
        T0,
        gamma=gamma,
        sigma=sigma,
        n_gradient=n_gradient,
        optimality=optimality
    )
    estimation = np.array(agent.identify(n_epochs)).squeeze()
    residual = estimation - A.numpy()
    return residual

if __name__ == '__main__':
    agent_index, criterion_index = int(str(sys.argv[1])[0]), int(str(sys.argv[1])[1])
    agent_ = [Active, Oracle][agent_index-1]
    optimality = ['A', 'D', 'E', 'L', 'T'][criterion_index-1]
    print(f'agent type {agent_index-1}, optimality {optimality}')
# parameters

    t_start = time.time()
    
    residuals = {}
    n_processes = mp.cpu_count()
    print(f'Running on {n_processes} processes')


    
    pool = mp.Pool(n_processes)
    residuals = np.zeros((n_samples, n_epochs+1, d, d))

    # ctx = mp.get_context('spawn')
    # with ctx.Pool(n_processes) as pool:
    sample_results = [pool.apply_async(identify_, args=(agent_, optimality,)) for _ in range(n_samples)]
    for sample_index, result in enumerate(sample_results):
        residuals[sample_index] = result.get()

    print(f'simulations completed')

    t_end = time.time()
    print(f'Experiment run in {t_end-t_start} seconds for {n_samples} samples')


    output_name = f'agent_{agent_index}_{optimality}_{n_samples}-samples_{n_gradient}-gradients_{n_epochs}-epochs'
    
    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(residuals, f)

