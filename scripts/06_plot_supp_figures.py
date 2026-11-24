import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import probplot
from src.logging_setup import setup_logger
from src.constants import PATHS, TAU_LIST, OMEGA_FULL, SEED
from src.amplitude_models import A_model_B, model_A_ansatz
from src.adjoint_sensitivity import hessian_approx, confidence_ellipse
from src.numerical_utils import ensure_dir

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
})
ensure_dir(PATHS["fig_supp"])
logger = setup_logger(PATHS["log"])

# S1 — sum-rule
fig, ax = plt.subplots(figsize=(6, 4))
svals = []
for om in OMEGA_FULL:
    amp = np.array([A_model_B(om, t) for t in TAU_LIST])
    svals.append(np.sum(amp ** 2))
ax.plot(OMEGA_FULL, svals, marker="o")
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel(r"$\sum A_k^2$")
ax.set_title("Figure S1: Sum-rule invariant")
fig.savefig(f"{PATHS['fig_supp']}/figS1.pdf")
plt.close(fig)

# S2 — Q-Q
np.random.seed(SEED)
x_fit = np.linspace(0, 1, 25)
p_true = np.array([0.2122, 0.4871, 0.0923, 0.1744, 0.0412, 0.3088, 0.1255, 0.0731])
y_true = model_A_ansatz(x_fit, p_true) + np.random.normal(0, 0.01956, size=25)
y_pred = model_A_ansatz(x_fit, p_true)
res = y_true - y_pred
fig, ax = plt.subplots(figsize=(6, 4))
probplot(res, plot=ax)
ax.set_title("Figure S2: Q-Q plot residuals")
fig.savefig(f"{PATHS['fig_supp']}/figS2.pdf")
plt.close(fig)

# S3 — confidence ellipse (first 2 params)
def obj(p):
    yhat = model_A_ansatz(x_fit, p)
    return np.sum((y_true - yhat) ** 2)

H = hessian_approx(obj, p_true[:2])
eigval, eigvec = confidence_ellipse(H)
fig, ax = plt.subplots(figsize=(5, 5))
theta = np.linspace(0, 2 * np.pi, 120)
ell = np.vstack([np.sqrt(eigval[0]) * np.cos(theta),
                 np.sqrt(eigval[1]) * np.sin(theta)])
ell = eigvec @ ell
ax.plot(ell[0, :] + p_true[0], ell[1, :] + p_true[1])
ax.scatter(p_true[0], p_true[1], c="red")
ax.set_title("Figure S3: Confidence ellipse")
fig.savefig(f"{PATHS['fig_supp']}/figS3.pdf")
plt.close(fig)

logger.info("Supplementary figures saved")