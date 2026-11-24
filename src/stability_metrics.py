import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from src.amplitude_models import A_k, sigma0_profile

def lyapunov_stability_check(mat):
    dim = mat.shape[0]
    Q = np.eye(dim)
    try:
        P = solve_continuous_lyapunov(mat.T, -Q)
        eig = np.linalg.eigvalsh(P)
        return np.all(eig > 0)
    except Exception:
        return False

def aic(loglik, n_param):
    return -2 * loglik + 2 * n_param

def adj_r2(y_true, y_pred, n_param):
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - n_param - 1)
    return adj, r2, ss_res, ss_tot

def sum_rule(amp_array):
    return np.sum(amp_array**2)