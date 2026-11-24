from src.consistency_check import run_cmp_consistency


def test_consistency_basic():
    pass_cnt, total, log = run_cmp_consistency()
    assert pass_cnt >= 30