import numpy as np
from scipy.integrate import solve_ivp
from src.constants import ODE_ATOL, ODE_RTOL

def lindbladian_rhs(t, rho, L, jump_ops):
    """
    Master equation: dρ/dt = -i [H, ρ] + sum_j ( L_j ρ L_j^† - 1/2 {L_j^† L_j, ρ} )
    rho: flattened density matrix
    """
    dim = int(np.sqrt(rho.shape[0]))
    rho_mat = rho.reshape(dim, dim)
    drho = -1j * (L @ rho_mat - rho_mat @ L)
    for op in jump_ops:
        term = op @ rho_mat @ op.conj().T
        term = term - 0.5 * (op.conj().T @ op @ rho_mat + rho_mat @ op.conj().T @ op)
        drho += term
    return drho.reshape(-1)

def evolve_lindbladian(L, jump_ops, rho0, t_span, t_eval=None):
    dim = rho0.shape[0]
    rho0_flat = rho0.reshape(-1)
    sol = solve_ivp(
        lambda t, rho: lindbladian_rhs(t, rho, L, jump_ops),
        t_span, rho0_flat, t_eval=t_eval,
        method="RK45", atol=ODE_ATOL, rtol=ODE_RTOL, dense_output=True
    )
    return sol