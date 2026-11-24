import numpy as np

def finite_diff_gradient(func, p, dp=1e-5):
    grad = np.zeros_like(p)
    for i in range(len(p)):
        p_up = p.copy()
        p_up[i] += dp
        p_dn = p.copy()
        p_dn[i] -= dp
        grad[i] = (func(p_up) - func(p_dn)) / (2 * dp)
    return grad

def hessian_approx(func, p, dp=1e-4):
    n = len(p)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            pp = p.copy(); pp[i] += dp; pp[j] += dp; fpp = func(pp)
            pm = p.copy(); pm[i] += dp; pm[j] -= dp; fpm = func(pm)
            mp = p.copy(); mp[i] -= dp; mp[j] += dp; fmp = func(mp)
            mm = p.copy(); mm[i] -= dp; mm[j] -= dp; fmm = func(mm)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * dp * dp)
            H[j, i] = H[i, j]
    return H

def confidence_ellipse(H, cov_scale=3.841):
    eigval, eigvec = np.linalg.eigh(H)
    eigval = np.maximum(eigval, 1e-12)
    return eigval, eigvec