import pandas as pd
from src.logging_setup import setup_logger
from src.constants import PATHS, TAU_LIST
from src.amplitude_models import boundary_min_criterion
from src.latex_export import table_to_latex
from src.numerical_utils import ensure_dir

ensure_dir(PATHS["latex"])
logger = setup_logger(PATHS["log"])

rows = []
for k in TAU_LIST:
    flag, sr, ar = boundary_min_criterion(k)
    rows.append({"$k$": k, r"$\sigma_0$ ratio": sr, r"$A_k$ ratio": ar, "Survive": flag})

df_tab = pd.DataFrame(rows)
table_to_latex(
    df_tab,
    "Boundary minimum classification",
    "threshold",
    f"{PATHS['latex']}/tab_threshold.tex",
)
logger.info("LaTeX tables exported")