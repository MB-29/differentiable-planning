import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad


class Information(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, A, T):
        d = A.shape[0]
        x = torch.zeros(d)
        X = torch.zeros(T, d)
        for t in range(T-1):
            x = A @ x + u[t]
            X[t+1, :] = x
            # plt.scatter(*x)
        M = X.T @ X
        # print(f'X = {X}')
        # print(f'M = {M}')
        # print(f'det M = {torch.det(M)}')
        # plt.pause(0
        
        loss = - torch.log(torch.det(M))
        ctx.save_for_backward(X, M, A)
        # return loss + 100* torch.sum(u**2)
        return loss 

    @staticmethod
    def backward(ctx, grad_output):
        X, M, A = ctx.saved_tensors
        T, d = X.shape
        lambd = torch.zeros(d)
        M_inv = torch.linalg.inv(M)
        L = torch.zeros(T, d)
        L[T-1] = 0
        for t in range(1, T):
            lambd += A @ lambd - M_inv @ X[T-1-t]
            L[T-2-t, :] = lambd
        grad_u = (grad_output.unsqueeze(0) @ L.reshape(1, d*T)).reshape(T, d)
        return grad_u, None, None
        
        


