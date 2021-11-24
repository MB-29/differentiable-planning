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

if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('task_id', type=int)
    # arg_parser.add_argument('-o', '--optimality', type=str, default='D')
    # arg_parser.add_argument('-c', '--certainty', action='store_true', default=False)
    # args = arg_parser.parse_args()
    # optimality = args.optimality
    # task_id = args.task_id
    # certainty = args.certainty
    #
    task_id = int(sys.argv[1])


    # for optimality in ['A', 'D', 'E', 'L']:
    full = torch.ones(d, dtype=torch.bool)
    partial = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    columns = partial if task_id%2 == 0 else full
    output = {}
    
    loss = np.zeros((n_steps, n_samples))
    error = np.zeros((n_steps, n_samples))
    X_data = torch.zeros(1, columns.sum())
    for sample_index in tqdm(range(n_samples)):
        # A = generate_random_A(d)
        controller = DiscreteController(
            A,
            B,
            T,
            gamma,
            sigma,
            optimality='E',
            columns=columns
            )
        loss_values, error_values = controller.plan(
            n_steps,
            batch_size,
            learning_rate=learning_rate,
            test='partial'
            )
        loss[:, sample_index] = loss_values
        error[:, sample_index] = error_values

    output['loss'] = loss_values
    output['error'] = error_values
    output['optimality'] = 'E'
    output['gamma'] = gamma
    output['sigma'] = sigma
    output['T0'] = T

    output_name = f'force_{columns.sum()}-columns_{n_samples}-samples_{n_steps}-gradients_{task_id}'

    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(output, f)
