"""Quick parameter estimation + absolute residuals for rfl_inla."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from scipy.stats import norm, lognorm
from scipy.optimize import minimize_scalar, minimize, dual_annealing
import warnings; warnings.filterwarnings('ignore')

# ── load helpers from rfl_inla.py without running main ──────────────
_src = open('rfl_inla.py', encoding='utf-8').read()
_cut = _src.split("if __name__ == '__main__':")[0]
exec(_cut, globals())

# ── parameter estimation ─────────────────────────────────────────────
print("Fitting LogNormal g + Normal f (n_grid=60, SA=300)...")
th, ll = heuristic_optimize(Y_r, S_r, cens_none, n_grid=60, sa_maxiter=300, seed=42)
b0, b1, ls, mu_d, lsd = th
sig = np.exp(ls); sig_d = np.exp(lsd)
print(f"b0={b0:.6f}  b1={b1:.6f}  sig={sig:.6f}")
print(f"mu_d={mu_d:.6f}  sig_d={sig_d:.6f}  E[D]={np.exp(mu_d+sig_d**2/2):.6f}")
print(f"ll={ll:.6f}")
print()

# ── absolute residuals ───────────────────────────────────────────────
# For each obs: find posterior mode D_hat_i, compute:
#   conditional residual  e_i = Y_i - (b0 + b1 log(S_i - D_hat_i))
#   absolute residual     |e_i|
#   standardised residual e_i / sigma
print(f"{'S_i':>6}  {'Y_i':>8}  {'D_hat':>7}  {'mu_hat':>8}  {'resid':>8}  {'abs_r':>7}  {'std_r':>7}")
print("-" * 60)

rows = []
for i in range(len(Y_r)):
    y, s = Y_r[i], S_r[i]
    res = minimize_scalar(
        lambda d: -_log_h(d, y, s, False, b0, b1, sig, mu_d, sig_d),
        bounds=(1e-6, s - 1e-4), method='bounded'
    )
    d_hat  = res.x
    mu_hat = b0 + b1 * np.log(s - d_hat)
    e_i    = y - mu_hat
    abs_r  = abs(e_i)
    std_r  = e_i / sig
    rows.append((s, y, d_hat, mu_hat, e_i, abs_r, std_r))
    print(f"{s:6.3f}  {y:8.4f}  {d_hat:7.5f}  {mu_hat:8.4f}  {e_i:+8.4f}  {abs_r:7.4f}  {std_r:+7.3f}")

print("-" * 60)
absr_arr = np.array([r[5] for r in rows])
e_arr    = np.array([r[4] for r in rows])
print(f"MAE  = {absr_arr.mean():.5f}")
print(f"MedAE= {np.median(absr_arr):.5f}")
print(f"RMSE = {np.sqrt((e_arr**2).mean()):.5f}")
print(f"Max |e| = {absr_arr.max():.5f}  at obs {absr_arr.argmax()} (S={rows[absr_arr.argmax()][0]:.3f})")

# ── save params for GitHub readme ───────────────────────────────────
print()
print("=== SUMMARY FOR README ===")
print(f"b0={b0:.4f}  b1={b1:.4f}  sigma={sig:.4f}")
print(f"mu_Delta={mu_d:.4f}  sigma_Delta={sig_d:.4f}  E[Delta]={np.exp(mu_d+sig_d**2/2):.4f}")
print(f"log-likelihood={ll:.4f}")
