import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch.multiprocessing as mp
import time
import sys

from utils import generate_random_A
from discrete_controller import DiscreteController

n_steps = 100
batch_size = 100
learning_rate = 0.1
n_samples = 100

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
gamma = 10
sigma = 0.1

T = 50


prior_mean = A.clone()
prior_mean[2:, :2] = 0


if __name__ == '__main__':
    task_id = int(sys.argv[1])
    prior = bool(task_id%2)

    prior_precision = torch.zeros(d, d, d)
    for j in range(d):
        prior_precision[j] = 1e7*torch.eye(d)
    prior_precision[2][:2, :2] = 0
    prior_precision[3][:2, :2] = 0


    # for optimality in ['A', 'D', 'E', 'L']:
    full = torch.ones(d, dtype=torch.bool)
    output = {}
    
    loss = np.zeros((n_steps, n_samples))
    error = np.zeros((n_steps, n_samples))
    for sample_index in tqdm(range(n_samples)):
        # A = generate_random_A(d)
        controller = DiscreteController(
            A,
            B,
            T,
            gamma,
            sigma,
            optimality='E',
            mean=prior_mean,
            precision=prior_precision
            )
        sample_loss, sample_error = controller.plan(
            n_steps,
            batch_size,
            learning_rate=learning_rate,
            test='partial',
            prior=prior
            )
        loss[:, sample_index] = sample_loss
        error[:, sample_index] = sample_error

    output['loss'] = loss
    output['error'] = error
    output['optimality'] = 'E'
    output['precision'] = prior_precision
    output['gamma'] = gamma
    output['sigma'] = sigma
    output['T0'] = T

    output_name = f'force_prior-{prior}_{n_samples}-samples_{n_steps}-gradients_{task_id}'

    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(output, f)
