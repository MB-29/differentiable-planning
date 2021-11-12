import numpy as np
import torch
import time
import pickle
import sys
from tqdm import tqdm

from agents import Active, Oracle, Random

T0 = 100
n_samples = 1
n_epochs = 8
gamma = 1
sigma = 0.1
n_gradient = 100

A = torch.tensor([

    [0.9, 1, 0, 0],
    [0, 0.9, 1, 0],
    [0, 0, 0.9, 1],
    [0, 0, 0, 0.9],
])
d = A.shape[0]
B = torch.eye(d)


if __name__ == '__main__':
    arg = sys.argv[1]
    # agent_index = int(str(arg)[0])
    # criterion_index = int(str(arg)[1])
    # job_index = int(str(arg)[2])
    # agent_ = [Active, Oracle][agent_index-1]
    # optimality = ['A', 'D', 'E', 'L', 'T'][criterion_index-1]
    # print(f'agent type {agent_index-1}, optimality {optimality}')
    optimality = 'L'
    agent_ = Oracle

    residuals = np.zeros((n_samples, n_epochs+1, d, d))
    for sample_index in tqdm(range(n_samples)):
        agent = agent_(
            A,
            B,
            T0,
            gamma=gamma,
            sigma=sigma,
            n_gradient=n_gradient,
            optimality=optimality
        )

        t_start = time.time()
        sample_estimations = np.array(agent.identify(n_epochs)).squeeze()
        t_end = time.time()

        sample_residuals = sample_estimations - A.numpy()
        residuals[sample_index, :, :, :] = sample_residuals
    


    output_name = f'{optimality}_{n_samples}-samples_{n_gradient}-gradients_{n_epochs}-epochs_{arg}'
    
    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(residuals, f)

