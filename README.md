# Differentiable active learning for system identification

## Usage


### Requirements
* Python 3
* torch
* numpy
* pytorch-lightning

Install the required packages with `pip install -r requirements.txt`.


### Use our module


```python
from agents import Active

# ... 

 agent = agent_contructor(
     A,                         # dynamics matrix              
     T0,                        # initial time horizon
     d,                         # state space dimension
     gamma,                     # gamma**2 is the energy
     sigma,                     # size of the noise
     n_gradient,                # number of gradient steps per epoch
     method                     # method
     )

# active learning for n_steps steps, evaluating on n_samples
estimations = agent.identify(n_epochs)
# estimations is a list of length n_steps+1 containing numpy arrays of shape (d, d)

```

## Example
  A script comparing our differentiable programming active learning approach versus a normally-distributed baseline is provided in the file `benchmark.py`.
  We obtained the following result.

### Approximation error versus epochs
![Different criteria](results/legend.png )
![Different criteria](results/oracles.png )
![Estimation at long time](results/long_time.png )
![Random matrices](results/random.png )

## References
Our algorithm is inspired by that of Wagenmaker *et al.* :

Andrew Wagenmaker, & Kevin Jamieson. (2020). Active Learning for Identification of Linear Dynamical Systems. 
