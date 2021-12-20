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
gamma = 10
sigma = 1
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

prior_mean = A.clone()
prior_mean[2:, :2] = 0

prior_precision = torch.zeros(d, d, d)
for j in range(d):
    prior_precision[j] = 1e7*torch.eye(d)
prior_precision[2][:2, :2] = 0
prior_precision[3][:2, :2] = 0

if __name__ == '__main__':
    task_id = int(sys.argv[1])
    print(f'{n_samples} samples, task {task_id}')
    partial = (task_id %2 == 0)

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
            mean=prior_mean,
            precision=prior_precision
        )

        sample_estimations = np.array(agent.identify(n_epochs)).squeeze()

        # sample_A_estimations = A.unsqueeze(0).expand(n_epochs+1, -1, -1).numpy()
        # sample_A_estimations[:, rows][:, :, columns] = sample_estimations

        # sample_residuals = sample_estimations - A[rows][:, columns].numpy()
        sample_residuals = sample_estimations - A.numpy()
        residuals[sample_index, :, :, :] = sample_residuals
    
    output = {
        'residuals': residuals,
        'gamma': gamma,
        'sigma': sigma,
        'T0': T0
    }
    output_name = f'id_force_{n_samples}-samples_{n_gradient}-gradients_{n_epochs}-epochs_{task_id}'
    
    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(output, f)

