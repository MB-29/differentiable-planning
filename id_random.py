import numpy as np
import torch
import time
import pickle
import sys
from tqdm import tqdm

from agents import Active, Oracle, Random
from utils import generate_random_A

T0 = 100
n_samples = 2
n_epochs = 7
gamma = np.sqrt(1000)
sigma = 1
n_gradient = 100
batch_size = 100

d = 5
m = 3


if __name__ == '__main__':
    arg = sys.argv[1]
    print(f'{n_samples} samples, arg {arg}')
    # agent_index = int(str(arg)[0])
    # criterion_index = int(str(arg)[1])
    # job_index = int(str(arg)[2])
    # agent_ = [Active, Oracle][agent_index-1]
    # optimality = ['A', 'D', 'E', 'L', 'T'][criterion_index-1]
    # print(f'agent type {agent_index-1}, optimality {optimality}')
    optimality = 'E'
    agent_ = Oracle

    residuals = np.zeros((n_samples, n_epochs+1, d, d))
    for sample_index in range(n_samples):
        print(f'sample {sample_index}')

        A = generate_random_A(d)
        B = torch.randn(d, m)
        agent = agent_(
            A,
            B,
            T0,
            gamma=gamma,
            sigma=sigma,
            batch_size=batch_size,
            n_gradient=n_gradient,
            optimality=optimality
        )
        sample_estimations = np.array(agent.identify(n_epochs)).squeeze()

        sample_residuals = sample_estimations - A.numpy()
        residuals[sample_index, :, :, :] = sample_residuals
    


    output_name = f'random_{optimality}-optimality_{n_samples}-samples_{n_gradient}-gradients_{n_epochs}-epochs_{arg}'
    
    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(residuals, f)

