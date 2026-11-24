from src.logging_setup import setup_logger
from src.constants import PATHS
from src.consistency_check import run_cmp_consistency
from src.numerical_utils import ensure_dir

ensure_dir(PATHS["report"])
logger = setup_logger(PATHS["log"])

pass_cnt, total, log = run_cmp_consistency()
report_text = "\n".join(log) + f"\n\nSummary: {pass_cnt}/{total} checks passed."

with open(f"{PATHS['report']}/pre_submit_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
logger.info(report_text)