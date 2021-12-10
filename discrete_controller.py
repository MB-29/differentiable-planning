from scipy.linalg import lstsq
import os



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch

from utils import criteria, estimate, estimate_batch, gramian, toeplitz
from adjoint import Evaluation


class DiscreteController:
    def __init__(self, A, B, T, gamma, sigma, mean=None, precision=None, x=None, optimality=''):
        super().__init__()
        self.T = T
        self.A = A
        self.B = B
        self.d, self.m = B.shape
        self.x = torch.zeros(self.d) if x is None else x

        self.mean = torch.zeros(self.d, self.d) if mean is None else mean
        precision = torch.zeros(self.d, self.d, self.d)
        for j in range(self.d):
            precision[j] = torch.eye(self.d)
        self.precision = torch.zeros(self.d, self.d) if precision is None else precision


        self.gamma = gamma
        self.sigma = sigma

        self.U = torch.randn(self.T, self.m, requires_grad=True)
        # self.U = torch.ones(self.T, self.m, requires_grad=True)

        self.criterion = criteria.get(optimality)

        self.gramian = gramian(A, T)

        self.covariates_matrix = toeplitz(A, T)


    def forward(self, x, stochastic=True):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.integration(x, self.A, U, stochastic), U
        # return self.integration(x, self.covariates_matrix, U, stochastic), U

    def integration_(self, x, covariates, U, stochastic):
        batch_size = x.shape[0]
        X = x.unsqueeze(1).expand(-1, self.T+1, -1).clone()
        control_input = (U@self.B.T).view(self.d*self.T)
        control_X = (covariates_matrix@control_input).view(self.T, self.d)
        X[:, 1:] += control_X.unsqueeze(0).expand(batch_size, -1, -1)
        
        if stochastic:
            W = self.sigma * torch.randn(self.T*self.d, batch_size)
            noise_X =  (self.covariates_matrix@W).reshape(batch_size, self.T, self.d)
            X[:, 1:] += noise_X
        return X
    def integration(self, x, A, U, stochastic):
        batch_size = x.shape[0]
        X = torch.zeros(batch_size, self.T+1, self.d)
        for t in range(self.T):
            u =  U[t, :] 
            x = (A @ x.T).T + self.B@u 
            if stochastic:
                noise = self.sigma * torch.randn_like(x)
                x += noise
            X[:, t+1, :] = x
        # print(f'played mean energy {(U**2).sum()/self.T}')
        return X

    def play(self, x, A, U):
        # print(f'played mean energy {(U**2).sum() / self.T}')
        energy_constraint = (torch.sum(U**2) / self.T <= (self.gamma**2)*1.1)
        assert energy_constraint, f'energy constraint not met : mean energy {torch.sum(U**2) / self.T}'
        covariates = toeplitz(A, self.T)
        # return self.integration(x, covariates, U, stochastic=True), U
        return self.integration(x, A, U, stochastic=True), U

    def play_control(self, x, A):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        return self.play(x, A, U)

    def play_random(self, x, A):
        U = torch.randn(self.T, self.m)
        U_normalized = self.gamma * np.sqrt(self.T) * U / torch.norm(U)
        return self.play(x, A, U_normalized)
        
    def plan(self, n_steps, batch_size, stochastic=True, learning_rate=0.1, test=None):
        if not stochastic:
            return self.plan_certainty(n_steps, batch_size, learning_rate, test)
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        loss_values = []
        error_values = []
        for step_index in range(n_steps):

            if test is not None:
                # and int(100*step_index/n_steps)%10 == 0:
                test_loss, error = self.test(test, batch_size)
                # test_loss, error = self.test_batch(batch_size)
                
                loss_values.append(test_loss)
                error_values.append(error.item())

            x = self.x.unsqueeze(0).expand(batch_size, self.d)
            X, U = self.forward(x, stochastic)
            S = torch.zeros(batch_size, 0)
            for row_index in range(self.d):
                certain_indices = torch.diag(self.precision[row_index]) >= float("Inf")
                prior_precision = self.precision[row_index,~certain_indices,~certain_indices].unsqueeze(0)
                sliced_X = X[:, :, ~certain_indices]
                fisher_matrix = sliced_X.permute(0, 2, 1)@sliced_X

                posterior_precision = prior_precision.expand(batch_size, -1, -1) + fisher_matrix
                eigenvalues = torch.linalg.eigvals(posterior_precision)
                assert eigenvalues.shape[0] == batch_size
                S = torch.cat((S, torch.real(eigenvalues)), dim=1)
            S, _ = torch.sort(S, descending=True)

            # print(f'{X_data.shape}, {X.shape}')
            # print(S)
            # print(S.min())
            loss = self.criterion(S, self.T)
            # print(f'loss {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.U.data = self.gamma *np.sqrt(self.T) * self.U / torch.norm(self.U)
            
        return loss_values, error_values

    def plan_certainty(self, n_steps, batch_size, learning_rate=0.1, test=None):
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        loss_values = []
        error_values = []
        for step_index in range(n_steps):

            if test is not None:
                # and int(100*step_index/n_steps)%10 == 0:
                test_loss, error = self.test(test, batch_size)
                # test_loss, error = self.test_batch(batch_size)
                
            
                loss_values.append(test_loss)
                error_values.append(error.item())

            x = torch.zeros(1, self.d)
            X, U = self.forward(x, False)
            X = X.squeeze()
            M = X.T @ X
            M += (self.sigma**2) * self.gramian

            S = torch.linalg.eigvals(M).unsqueeze(0)
            S, _ = torch.sort(torch.real(S), descending=True)
            # print(S.min())
            loss = self.criterion(S, self.T)
            # print(f'loss {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.U.data = self.gamma *np.sqrt(self.T) * self.U / torch.norm(self.U)
            
        return loss_values, error_values

    def plan_adjoint(self, n_steps, batch_size, stochastic, learning_rate=0.1, test=False):
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        loss_values = []
        error_values = []
        for step_index in range(n_steps):

            if test:
                test_loss, error = self.test_batch(batch_size)
                # print(f'test loss {test_loss.item()}')
            
                loss_values.append(test_loss.item())
                error_values.append(error.item())

            U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
            loss = Evaluation.apply(self.A, self.B, self.U, self.T, self.sigma)
            # print(f'training loss {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.U.data = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)

            
        return loss_values, error_values
    

    def test(self, test_type, batch_size):
        with torch.no_grad():
            x = torch.zeros(1, self.d)
            X, U = self.play_control(x, self.A)
            # X, U = self.forward(x, False)
            S = torch.linalg.svdvals(X[:, :-1])
            if test_type == 'criterion':
                test_loss = self.criterion(S, self.T)
            elif test_type == 'sv':
                test_loss = [S[0, -1], S[0, 0]]
            elif test_type == 'partial':

                test_loss = torch.linalg.norm(X[:, -1, :2])

                X_tilde = X.squeeze()[:-1, :2]
                X_bar = X.squeeze()[:-1, 2:]
                A_bar = self.A[2:, 2:]
                A_tilde = self.A[2:, :2]
                Y = (X.squeeze()[1:, :] - U@self.B.T)[:, 2:] - X_bar@A_bar.T
                solution ,_, _, _ = lstsq(X_tilde, Y)
                estimation = solution.T
                # print(f'estimation {estimation}')
                # print(f'A_tilde {A_tilde}')
                error = np.linalg.norm(estimation - A_tilde.numpy())
                return test_loss, error
            # M = X.permute(0, 2, 1) @ X.permute(0, 1, 2)
            # test_loss = - torch.log(torch.det(M)).mean()

            A_hat = estimate(X.squeeze(), U)
            error = torch.linalg.norm(A_hat - self.A)
            energy = torch.sum(U**2)/ self.T
            # print(f'X.shape {X.shape}, energy {energy}, A = {self.A}, A_hat = {A_hat}')
            # print(f'error {error}')
            return test_loss, error

    def test_batch(self, batch_size):
        with torch.no_grad():
            x = torch.zeros(batch_size, self.d)
            X, U = self.play_control(x, self.A)
            energy_constraint = (torch.sum(U**2) / self.T <= (self.gamma**2)*1.1)
            assert energy_constraint, f'energy constraint not met : mean energy {torch.sum(U**2) / self.T}'
            # X, U = self.forward(x, True)
            A_hat = estimate_batch(X, U.unsqueeze(0))
            error = torch.linalg.norm(A_hat - self.A, dim=(1,2)).mean()
            # print(f'test error {error}')

            S = torch.linalg.svdvals(X[:, :-1, :])
            # test_loss = self.criterion(S, self.T)
            test_loss = S[:, -1].mean()
            # M = X.permute(0, 2, 1) @ X.permute(0, 1, 2)
            # test_loss = - torch.log(torch.det(M)).mean()


            return test_loss, error

        
