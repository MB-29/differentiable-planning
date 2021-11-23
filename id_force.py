import numpy as np
import torch
import time
import pickle
import sys
from tqdm import tqdm

from agents import Active, Oracle, Random

T0 = 20
n_samples = 100
n_epochs = 3
gamma = 1
sigma = 0.1
n_gradient = 100
batch_size = 100
dt = 0.1

phi = 0.1
u = np.cos(phi)
v = np.sin(phi)

A = torch.tensor([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [-0.2*u, 0.2*v, 1-dt, 0],
    [-0.5*v, -0.5*u, 0, 1-dt],
],
dtype=torch.float)
B = torch.tensor([
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt]
])
d = 4
L = torch.linalg.eigvals(A)
print(L)
print(torch.abs(L).max())

rows = torch.ones(d, dtype=torch.bool)
columns = torch.ones(d, dtype=torch.bool)
rows = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
columns = torch.tensor([1, 1, 0, 0], dtype=torch.bool)


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
        agent = agent_(
            A,
            B,
            T0, 
            gamma=gamma,
            sigma=sigma,
            batch_size=batch_size,
            n_gradient=n_gradient,
            optimality=optimality,
            rows=rows,
            columns=columns
        )

        sample_estimations = np.array(agent.identify(n_epochs)).squeeze()

        # sample_A_estimations = A.unsqueeze(0).expand(n_epochs+1, -1, -1).numpy()
        # sample_A_estimations[:, rows][:, :, columns] = sample_estimations

        # sample_residuals = sample_estimations - A[rows][:, columns].numpy()
        sample_residuals = sample_estimations - A.numpy()
        residuals[sample_index, :, :, :] = sample_residuals
    

    output_name = f'id_force_partial_{n_samples}-samples_{n_gradient}-gradients_{n_epochs}-epochs_{arg}'
    
    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(residuals, f)

