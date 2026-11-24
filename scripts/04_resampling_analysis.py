import numpy as np
import pandas as pd
from src.logging_setup import setup_logger
from src.constants import PATHS, SEED, N_FIT_POINTS
from src.amplitude_models import model_A_ansatz
from src.resampling import run_bootstrap, jackknife_delete_d
from src.stability_metrics import adj_r2
from src.numerical_utils import ensure_dir

np.random.seed(SEED)
ensure_dir(PATHS["csv"])
logger = setup_logger(PATHS["log"])

x_fit = np.linspace(0, 1, N_FIT_POINTS)
p_true = np.array([0.2122, 0.4871, 0.0923, 0.1744, 0.0412, 0.3088, 0.1255, 0.0731])
y_true = model_A_ansatz(x_fit, p_true) + np.random.normal(0, 0.01956, size=N_FIT_POINTS)
y_pred = model_A_ansatz(x_fit, p_true)

def stat_r2(y):
    adj, _, _, _ = adj_r2(y, y_pred, 8)
    return adj

boot_ci, boot_mean, boot_samples = run_bootstrap(y_true, y_pred, stat_r2)
jack_ci, jack_mean = jackknife_delete_d(y_true, y_pred, stat_r2)

df_resamp = pd.DataFrame({
    "method": ["bootstrap", "jackknife"],
    "mean": [boot_mean, jack_mean],
    "ci_low": [boot_ci[0], jack_ci[0]],
    "ci_high": [boot_ci[1], jack_ci[1]],
})
df_resamp.to_csv(f"{PATHS['csv']}/resampling_stats.csv", index=False)
logger.info("Resampling analysis complete")