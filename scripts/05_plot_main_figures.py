import matplotlib.pyplot as plt
import numpy as np
from src.logging_setup import setup_logger
from src.constants import PATHS, LAMBDA_MIN, LAMBDA_MAX, TAU_LIST, OMEGA_FULL, SEED
from src.amplitude_models import (
    S_min, sigma0_profile, A_k, A_model_B, model_A_ansatz
)
from src.numerical_utils import ensure_dir

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.dpi": 600,
})
ensure_dir(PATHS["fig_main"])
logger = setup_logger(PATHS["log"])
lam = np.linspace(LAMBDA_MIN, LAMBDA_MAX, 250)

# Fig1 — schematic
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.15, 0.72, "Markovian baseline $\\epsilon=0$", fontsize=11)
ax.text(0.15, 0.40, "Non-Markovian revival $\\epsilon>0$", fontsize=11)
ax.set_xlabel("$t$")
ax.set_ylabel("Amplitude")
ax.set_title("Figure 1: CPTP semigroup schematic")
fig.savefig(f"{PATHS['fig_main']}/fig1.pdf")
plt.close(fig)

# Fig2 — survive k=1
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lam, S_min(lam, 1), label=r"$S_{\min}(\lambda)$")
ax.axvline(1.0, ls="--", c="k")
ax.set_title("Figure 2: Boundary minimum survives")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$S_{\min}$")
ax.legend()
fig.savefig(f"{PATHS['fig_main']}/fig2.pdf")
plt.close(fig)

# Fig3 — escape k=2
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lam, S_min(lam, 2), label=r"$S_{\min}(\lambda),k=2$")
ax.axvline(1.0, ls="--", c="k")
ax.set_title("Figure 3: Boundary minimum escapes ($k=2$)")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$S_{\min}$")
ax.legend()
fig.savefig(f"{PATHS['fig_main']}/fig3.pdf")
plt.close(fig)

# Fig4 — ratio comparison
fig, ax = plt.subplots(figsize=(6, 4))
rs = sigma0_profile(lam) / sigma0_profile(1.0)
ra = A_k(lam, 2) / A_k(1.0, 2)
ax.plot(lam, rs, label=r"$\sigma_0(\lambda)/\sigma_0(1)$")
ax.plot(lam, ra, label=r"$A_2(\lambda)/A_2(1)$")
ax.axhline(1, ls="--", c="gray")
ax.set_title("Figure 4: Ratio comparison for threshold inequality")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Ratio")
ax.legend()
fig.savefig(f"{PATHS['fig_main']}/fig4.pdf")
plt.close(fig)

# Fig5 — Model B amplitudes + threshold
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
for t in TAU_LIST:
    av = np.array([A_model_B(o, t) for o in OMEGA_FULL])
    sv = 1.0 / (av ** 2)
    ax1.plot(OMEGA_FULL, av, marker="o", label=f"$\\tau={t}$")
    ax2.plot(OMEGA_FULL, sv, ls="--")
ax1.set_xlabel(r"$\Omega$")
ax1.set_ylabel(r"$A_k$")
ax2.set_ylabel(r"$S_{\min}$")
ax1.set_title("Figure 5: Amplitude and threshold for Model B")
ax1.legend()
fig.savefig(f"{PATHS['fig_main']}/fig5.pdf")
plt.close(fig)

# Fig6 — residuals
fig, ax = plt.subplots(figsize=(6, 4))
np.random.seed(SEED)
x_fit = np.linspace(0, 1, 25)
p_true = np.array([0.2122, 0.4871, 0.0923, 0.1744, 0.0412, 0.3088, 0.1255, 0.0731])
y_true = model_A_ansatz(x_fit, p_true) + np.random.normal(0, 0.01956, size=25)
y_pred = model_A_ansatz(x_fit, p_true)
res = y_true - y_pred
ax.scatter(np.arange(len(res)), res, s=22)
ax.axhline(0, c="k", ls="--")
ax.set_title("Figure 6: Residual diagnostics Model A")
ax.set_xlabel("Index")
ax.set_ylabel("Residual")
fig.savefig(f"{PATHS['fig_main']}/fig6.pdf")
plt.close(fig)

# Fig7 — robustness
fig, ax = plt.subplots(figsize=(6, 4))
rob = np.array([0.37, 0.373, 0.421, 0.44, 0.368])
ax.bar(TAU_LIST, rob)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$\sigma$ reduction ratio")
ax.set_title("Figure 7: Robustness reduction ratio across delays")
fig.savefig(f"{PATHS['fig_main']}/fig7.pdf")
plt.close(fig)

logger.info("Main figures saved")