import numpy as np
import torch
import time
import pickle
import sys
from tqdm import tqdm
import argparse

from utils import generate_random_A
from discrete_controller import DiscreteController

T0 = 100
n_samples = 200
gamma = 10
sigma = 1
n_gradient = 300
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
    task_id = sys.argv[1]
    arg_1 = int(str(task_id)[0])
    arg_2 = int(str(task_id)[1])
    optimality = ['A', 'D', 'E', 'L', 'T'][arg_1-1]
    certainty = (arg_2%2 == 0)

    print(f'{n_samples} samples, task {task_id}, optimality {optimality}, certainty={certainty}')

    output = {}
    for sample_index in tqdm(range(n_samples)):
        print(f'sample {sample_index}')


        A = generate_random_A(d)
        loss = np.zeros((n_gradient, n_samples))
        error = np.zeros((n_gradient, n_samples))
        for sample_index in tqdm(range(n_samples)):
            A = generate_random_A(d)
            controller = DiscreteController(
                A, B, T0, gamma, sigma, optimality=optimality)
            loss_values, error_values = controller.plan(
                n_gradient,
                batch_size,
                learning_rate=learning_rate,
                certainty=certainty,
                test=True
            )
            loss[:, sample_index] = loss_values
            error[:, sample_index] = error_values

    output['loss'] = loss_values
    output['error'] = error_values
    output['optimalitiy'] = optimality
    output['certainty'] = certainty

    


    output_name = f'{optimality}-optimality_{n_samples}-samples_{n_gradient}-gradients_{task_id}'
    output_path = f'output/{output_name}'
    
    with open(f'{output_path}.pkl', 'wb') as f:
        pickle.dump(output, f)

