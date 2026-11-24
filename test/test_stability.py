import numpy as np
from src.stability_metrics import lyapunov_stability_check


def test_lyapunov_stable():
    A = np.array([[-1.0, 0.2], [0.2, -1.0]])
    assert lyapunov_stability_check(A) is True