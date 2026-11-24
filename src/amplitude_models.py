import numpy as np
from scipy.interpolate import interp1d
from src.constants import SIGMA0_REF_RATIO, A2_REF_RATIO

def sigma0_profile(lam):
    return np.interp(lam, [0.7, 1.0], [SIGMA0_REF_RATIO, 1.0])

def A_k(lam, k):
    if k == 2:
        a1 = 0.01837
        a07 = 0.04529
        return np.interp(lam, [0.7, 1.0], [a07, a1])
    else:
        return np.interp(lam, [0.7, 1.0], [0.05, 0.08])

def S_min(lam, k, alpha=1.0):
    s0 = sigma0_profile(lam)
    ak = A_k(lam, k)
    return (alpha**2) * (s0**2) / (ak**2)

def A_model_B(omega, tau):
    anchor = {
        1: np.array([0.63, 0.63]),
        2: np.array([2.47, 2.47]),
        3: np.array([0.63, 0.63]),
        4: np.array([0.58, 0.58]),
        5: np.array([0.56, 0.56])
    }
    f = interp1d([0.35, 0.60], anchor[tau], kind="cubic", fill_value="extrapolate")
    return f(omega)

def model_A_ansatz(x, p):
    p1, p2, p3, p4, p5, p6, p7, p8 = p
    return p1 * np.exp(-p2 * x) + p3 * np.cos(p4 * x) + p5 * np.sin(p6 * x) + p7 * x + p8

def boundary_min_criterion(k):
    s_ratio = sigma0_profile(0.7) / sigma0_profile(1.0)
    a_ratio = A_k(0.7, k) / A_k(1.0, k)
    survive = s_ratio > a_ratio
    return survive, s_ratio, a_ratio