import numpy as np
import pandas as pd
from src.logging_setup import setup_logger
from src.constants import SEED, ODE_T_MAX, PATHS
from src.numerical_utils import ensure_dir, save_cache
from src.lindbladian_evolve import evolve_lindbladian

np.random.seed(SEED)
ensure_dir(PATHS["cache"])
ensure_dir(PATHS["csv"])
logger = setup_logger(PATHS["log"])

dim = 2
H = np.array([[0.2, 0.1], [0.1, -0.2]], dtype=complex)
L1 = np.array([[0.0, 0.6], [0.0, 0.0]], dtype=complex)
jump_ops = [L1]
rho0 = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
t_eval = np.linspace(0, ODE_T_MAX, 600)

sol = evolve_lindbladian(H, jump_ops, rho0, (0, ODE_T_MAX), t_eval=t_eval)
rho_t = sol.y.T
df_evo = pd.DataFrame({"t": sol.t, "rho00": rho_t[:, 0].real, "rho11": rho_t[:, 3].real})
df_evo.to_csv(f"{PATHS['csv']}/lindbladian_evolution.csv", index=False)
save_cache(sol, f"{PATHS['cache']}/lind_sol.pkl")
logger.info("Lindbladian evolution finished")