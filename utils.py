import torch
from scipy.linalg import lstsq

def A_criterion(S, T):
    return T * (1/S**2).sum(dim=1).mean()
def D_criterion(S, T):
    return -torch.log(torch.prod(S, dim=1).mean())
def L_criterion(S, T):
    return -torch.sum(torch.log(S), dim=1).mean()
def E_criterion(S, T):
    return - S[:, -1].mean()
def T_criterion(S, T):
    return - (1/T) * (S**2).mean(dim=1).mean()


criteria = {
    'A': A_criterion,
    'D': D_criterion,
    'E': E_criterion,
    'L': L_criterion,
    'T': T_criterion
}


def estimate(X, U):
    Y = X[1:, :] - U
    solution, _, _, _ = lstsq(X[:-1, :], Y)
    estimation = solution.T
    return torch.tensor(estimation)

def estimate_batch(X, U):
    Y = X[:, 1:, :] - U
    estimation = torch.linalg.lstsq(X[:, :-1, :], Y).solution.permute(0, 2, 1)
    return estimation

def generate_random_A(d):
    M = torch.randn(d, d)
    eigenvals = torch.linalg.eigvals(M)
    rho = torch.abs(eigenvals).max()
    return M / rho

def gramian(A, T):
    matrix = 0
    iterate = torch.eye(*A.size())
    for t in range(T):
        matrix += (T-t)*iterate @ iterate.T
        iterate = A@iterate
    return matrix

def toeplitz(A, T):
    d, _ = A.shape
    gamma = torch.zeros(T*d, T*d)
    iterate = torch.eye(d)
    for t in range(T):
        for i in range(t, T):
            j = i - t
            gamma[d*i:d*(i+1), d*j: d*(j+1)] = iterate
        iterate = A@iterate
    return gamma


