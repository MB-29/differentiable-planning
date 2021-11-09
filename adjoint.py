import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad


class Evaluation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, U, T, sigma):
        d, m = B.shape
        x = torch.zeros(d)
        X = torch.zeros(T+1, d)
        for t in range(T):
            x = A @ x + B@U[t]
            X[t+1, :] = x
            # plt.scatter(*x)
        M = (1/sigma) * X.T @ X
        loss = - torch.log(torch.det(M))
        # print(f'X = {X}')
        # print(f'M = {M}')
        # print(f'det M = {torch.det(M)}')
        # plt.pause(0

        ctx.save_for_backward(X, M, A, B, U)
        # return loss + 100* torch.sum(u**2)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        X, M, A, B, U = ctx.saved_tensors
        T, m = U.shape
        lambd = torch.zeros(m)
        M_inv = torch.linalg.inv(M)
        L = torch.zeros(T, m)
        for t in range(T):
            lambd = A @ lambd - B@M_inv @ X[T-t]
            L[T-1-t, :] = lambd
        # print(L)
        grad_u = grad_output * L
        return None, None, grad_u, None, None
