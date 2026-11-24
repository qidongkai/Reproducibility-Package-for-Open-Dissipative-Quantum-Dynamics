import numpy as np
from scipy.optimize import minimize
from src.amplitude_models import model_A_ansatz
from src.constants import FIT_N_STARTS, FIT_REG_LAMBDA

def loss_reg(p, x, y, reg_lambda):
    yhat = model_A_ansatz(x, p)
    mse = np.sum((y - yhat) ** 2)
    reg = reg_lambda * np.sum(p ** 2)
    return mse + reg

def multi_start_lbfgs(x, y):
    best_loss = np.inf
    best_p = None
    bounds = [(-2, 2)] * 8
    for _ in range(FIT_N_STARTS):
        p0 = np.random.randn(8) * 0.25
        res = minimize(
            lambda p: loss_reg(p, x, y, FIT_REG_LAMBDA),
            p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 12000}
        )
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_p = res.x
    return best_p, best_loss