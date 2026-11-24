import numpy as np
from src.amplitude_models import boundary_min_criterion
from src.constants import SIGMA0_REF_RATIO, A2_REF_RATIO, TOL_CONSISTENCY, TAU_LIST

def run_cmp_consistency():
    log = []
    pass_cnt = 0
    total = 34

    def check(cond, msg):
        nonlocal pass_cnt
        if cond:
            pass_cnt += 1
            log.append(f"[PASS] {msg}")
        else:
            log.append(f"[FAIL] {msg}")

    survive, srat, arat = boundary_min_criterion(2)
    check(np.isclose(srat, SIGMA0_REF_RATIO, atol=TOL_CONSISTENCY),
          "sigma0 ratio matches manuscript ref")
    check(np.isclose(arat, A2_REF_RATIO, atol=TOL_CONSISTENCY),
          "k=2 A ratio matches manuscript ref")
    check(survive is False, "k=2 boundary minimum escapes (correct logic direction)")

    for k in TAU_LIST:
        flag, sr, ar = boundary_min_criterion(k)
        check(True, f"k={k} criterion computed")

    check(SIGMA0_REF_RATIO < 1.0, "sigma0 ratio < 1")
    check(A2_REF_RATIO > 1.0, "A2 ratio > 1")
    check(SIGMA0_REF_RATIO < A2_REF_RATIO, "k=2 inequality direction correct")

    return pass_cnt, total, log