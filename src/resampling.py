import numpy as np
from joblib import Parallel, delayed
from src.constants import BOOTSTRAP_JOBS, BOOTSTRAP_N, JACKKNIFE_D

def bootstrap_sample(y, yhat, fn):
    resid = y - yhat
    idx = np.random.randint(0, len(y), size=len(y))
    yboot = yhat + resid[idx]
    return fn(yboot)

def run_bootstrap(y, yhat, stat_fn):
    out = Parallel(n_jobs=BOOTSTRAP_JOBS)(
        delayed(bootstrap_sample)(y, yhat, stat_fn) for _ in range(BOOTSTRAP_N)
    )
    out = np.array(out)
    return np.quantile(out, [0.025, 0.975]), np.mean(out), out

def jackknife_delete_d(y, yhat, stat_fn):
    n = len(y)
    out = []
    for i in range(0, n - JACKKNIFE_D + 1):
        mask = np.ones(n, dtype=bool)
        mask[i:i + JACKKNIFE_D] = False
        ysub = y[mask]
        out.append(stat_fn(ysub))
    out = np.array(out)
    return np.quantile(out, [0.025, 0.975]), np.mean(out)