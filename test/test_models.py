import numpy as np
from src.amplitude_models import boundary_min_criterion, S_min, A_k


def test_k2_criterion():
    survive, srat, arat = boundary_min_criterion(2)
    assert survive is False
    assert np.isclose(srat, 0.62, atol=1e-4)
    assert np.isclose(arat, 2.4658, atol=1e-4)


def test_Smin_positive():
    for k in [1, 2, 3, 4, 5]:
        s = S_min(0.7, k)
        assert s > 0