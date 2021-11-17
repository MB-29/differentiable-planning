import numpy as np
import torch
import time
import pickle
import sys
from tqdm import tqdm
import argparse

from utils import generate_random_A
from discrete_controller import DiscreteController

T0 = 10
n_samples = 2
gamma = 1
sigma = 0.5
n_gradient = 2
batch_size = 100
d = 4
B = torch.eye(d)
learning_rate = 0.1



if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('task_id', type=int)
    # arg_parser.add_argument('-o', '--optimality', type=str, default='D')
    # arg_parser.add_argument('-c', '--certainty', action='store_true', default=False)
    # args = arg_parser.parse_args()
    # optimality = args.optimality
    # task_id = args.task_id
    # certainty = args.certainty        
    task_id = int(sys.argv[1])
    # arg_1 = int(str(task_id)[0])
    # arg_2 = int(str(task_id)[1])
    # optimality = ['A', 'D', 'E', 'L', 'T'][arg_1-1]
    optimality = 'E'
    stochastic = (task_id%2 == 0)

    print(f'{n_samples} samples, task {task_id}, optimality {optimality}, stochastic={stochastic}')
    print(f'gamma {gamma}, sigma={sigma}')

    output = {}
    loss_values = np.zeros((n_samples, n_gradient))
    error_values = np.zeros((n_samples, n_gradient))
    for sample_index in tqdm(range(n_samples)):
        print(f'sample {sample_index}')

        A = generate_random_A(d)
        controller = DiscreteController(
            A, B, T0, gamma, sigma, optimality=optimality)
        sample_loss, sample_error = controller.plan(
            n_gradient,
            batch_size,
            learning_rate=learning_rate,    
            stochastic=stochastic,
            test=True
        )
        loss_values[sample_index, :] = sample_loss
        error_values[sample_index, :] = sample_error

    output['loss'] = loss_values
    output['error'] = error_values
    output['optimality'] = optimality
    output['stochastic'] = stochastic
    output['gamma'] = gamma
    output['sigma'] = sigma
    


    output_name = f'{optimality}-optimality_{n_samples}-samples_{n_gradient}-gradients_{task_id}'
    output_path = f'output/{output_name}'
    
    with open(f'{output_path}.pkl', 'wb') as f:
        pickle.dump(output, f)

