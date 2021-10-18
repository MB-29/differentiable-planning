import torch

def A_criterion(S, T):
    return T * (1/S**2).sum(dim=1)
def D_criterion(S, T):
    return -torch.sum(torch.log(S), dim=1)
def E_criterion(S, T):
    return - S[:, -1]
def T_criterion(S, T):
    return - (1/T) * S.mean(dim=1)


criteria = {
    'A': A_criterion,
    'D': D_criterion,
    'E': E_criterion,
    'T': T_criterion
}
