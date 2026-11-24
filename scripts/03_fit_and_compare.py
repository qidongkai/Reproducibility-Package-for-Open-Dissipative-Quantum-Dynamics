import numpy as np
import pandas as pd
from src.logging_setup import setup_logger
from src.constants import PATHS, N_FIT_POINTS, SEED
from src.amplitude_models import model_A_ansatz
from src.fitting_engine import multi_start_lbfgs
from src.stability_metrics import adj_r2, aic
from src.numerical_utils import ensure_dir

np.random.seed(SEED)
ensure_dir(PATHS["csv"])
logger = setup_logger(PATHS["log"])

x_fit = np.linspace(0, 1, N_FIT_POINTS)
p_true = np.array([0.2122, 0.4871, 0.0923, 0.1744, 0.0412, 0.3088, 0.1255, 0.0731])
y_true = model_A_ansatz(x_fit, p_true) + np.random.normal(0, 0.01956, size=len(x_fit))

p_fit, loss = multi_start_lbfgs(x_fit, y_true)
y_pred_full = model_A_ansatz(x_fit, p_fit)

# nested model: only first 5 params active, remaining zeros
p_nested = p_fit.copy()
p_nested[5:] = 0.0
y_pred_nested = model_A_ansatz(x_fit, p_nested)

adj_full, r2_full, rss_full, tss = adj_r2(y_true, y_pred_full, 8)
adj_nested, r2_nested, rss_nested, _ = adj_r2(y_true, y_pred_nested, 5)

logL_full = -0.5 * len(x_fit) * np.log(rss_full / len(x_fit))
aic_full = aic(logL_full, 8)
logL_nested = -0.5 * len(x_fit) * np.log(rss_nested / len(x_fit))
aic_nested = aic(logL_nested, 5)

df_fit = pd.DataFrame({
    "x": x_fit,
    "y_true": y_true,
    "y_pred_full": y_pred_full,
    "y_pred_nested": y_pred_nested,
})
df_fit.to_csv(f"{PATHS['csv']}/modelA_fit.csv", index=False)

df_stats = pd.DataFrame({
    "model": ["full(8p)", "nested(5p)"],
    "adj_R2": [adj_full, adj_nested],
    "AIC": [aic_full, aic_nested],
    "RSS": [rss_full, rss_nested],
})
df_stats.to_csv(f"{PATHS['csv']}/fit_stats.csv", index=False)
logger.info("Fitting finished")