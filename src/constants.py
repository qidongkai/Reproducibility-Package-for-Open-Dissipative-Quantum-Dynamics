import yaml
from pathlib import Path

cfg_path = Path(__file__).parent.parent / "config/config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]
ALPHA = cfg["alpha"]
LAMBDA_MIN = cfg["lambda_min"]
LAMBDA_MAX = cfg["lambda_max"]
N_LAMBDA = cfg["n_lambda"]
TAU_LIST = cfg["tau_list"]
OMEGA_ANCHORS = cfg["omega_anchors"]
OMEGA_FULL = cfg["omega_full"]
N_FIT_POINTS = cfg["n_fit_points"]
SIGMA0_REF_RATIO = cfg["sigma0_ref_ratio"]
A2_REF_RATIO = cfg["a2_ref_ratio"]
BOOTSTRAP_N = cfg["bootstrap_n"]
JACKKNIFE_D = cfg["jackknife_d"]
BOOTSTRAP_JOBS = cfg["bootstrap_jobs"]
FIT_N_STARTS = cfg["fit_n_starts"]
FIT_REG_LAMBDA = cfg["fit_reg_lambda"]
ODE_T_MAX = cfg["ode_t_max"]
ODE_ATOL = cfg["ode_atol"]
ODE_RTOL = cfg["ode_rtol"]
TOL_CONSISTENCY = cfg["tolerance_consistency"]
PATHS = cfg["paths"]