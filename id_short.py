import numpy as np
import torch
import time
import pickle
import sys
from tqdm import tqdm

from agents import Active, Oracle, Random
from utils import generate_random_A

n_samples = 5

T0 = 100
n_epochs = 3
gamma = 5
sigma = 1
n_gradient = 1000
batch_size = 100

d = 4
m = 4
optimality = 'E'
rho = 0.5

B = torch.eye(m)

mean = torch.zeros(d, d)
precision = torch.zeros(d, d, d)
# for j in range(d):
#     precision[j] = torch.eye(d)

if __name__ == '__main__':
    arg = sys.argv[1]   
    print(f'{n_samples} samples, arg {arg}')
    # agent_index = int(str(arg)[0])
    # criterion_index = int(str(arg)[1])
    # job_index = int(str(arg)[2])
    # agent_ = [Active, Oracle][agent_index-1]
    # optimality = ['A', 'D', 'E', 'L', 'T'][criterion_index-1]
    # print(f'agent type {agent_index-1}, optimality {optimality}')

    agent_ = Active
    # agent_ = Random

    error_values = np.zeros((n_samples, n_epochs+1))
    time_values = []
    for sample_index in range(n_samples):
        print(f'sample {sample_index}')

        A = rho * generate_random_A(d) 
        agent = agent_(
            A,
            B,
            T0,
            gamma=gamma,
            sigma=sigma,
            batch_size=batch_size,
            n_gradient=n_gradient,
            optimality=optimality,
            mean=mean,
            precision=precision
        )
        start_time = time.time()
        sample_estimations = np.array(agent.identify(n_epochs)).squeeze()
        end_time = time.time()

        time_values.append(end_time - start_time)

        sample_residuals = sample_estimations - A.numpy()
        sample_error = np.linalg.norm(sample_residuals, axis=(1, 2))
        error_values[sample_index, :] = sample_error
    
    output={'op': error_values, 'time':time_values, 'rho':rho, 'sigma': sigma, 'gamma':gamma}

    output_name = f'oracle-random_{optimality}-optimality_{n_samples}-samples_{n_gradient}-gradients_{n_epochs}-epochs_{arg}'
    
    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(output, f)

