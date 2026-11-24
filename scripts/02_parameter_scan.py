import yaml
import numpy as np
import pandas as pd
from src.logging_setup import setup_logger
from src.constants import PATHS, LAMBDA_MIN, LAMBDA_MAX, N_LAMBDA
from src.amplitude_models import S_min
from src.numerical_utils import ensure_dir

ensure_dir(PATHS["csv"])
logger = setup_logger(PATHS["log"])
with open("config/param_scan.yaml", "r") as f:
    scan_cfg = yaml.safe_load(f)

lam_sweep = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_LAMBDA)
rows = []
for sigma_scale in scan_cfg["sigma0_scales"]:
    for k in scan_cfg["k_scan"]:
        for lam in lam_sweep:
            s = S_min(lam, k, alpha=sigma_scale)
            rows.append({"sigma_scale": sigma_scale, "k": k, "lam": lam, "S_min": s})

df_scan = pd.DataFrame(rows)
df_scan.to_csv(f"{PATHS['csv']}/parameter_sweep.csv", index=False)
logger.info("Parameter sweep complete")