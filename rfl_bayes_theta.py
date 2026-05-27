"""
rfl_bayes_theta.py  --  Direction B: Posterior predictive ASSE
================================================================
Replace plugin estimate ASSE(theta_hat) with posterior integration:

  ASSE_bayes = E_{theta ~ p(theta|y)} [ Sigma_i |y-hat_i(theta) - y_i| ]

Approach: Laplace approximation to the posterior
  p(theta|y) approx N(theta_hat, H_obs^{-1})

where H_obs = -d^2 logL / dtheta^2  (observed Fisher information)

Steps:
  1. Start from published MLE:  theta_hat from rfl_sev_inla.py  (Laplace GH)
  2. Numerical Hessian of Laplace GH loglik at theta_hat
  3. Sample M=2000 theta^m ~ N(theta_hat, H_obs^{-1})
  4. For each theta^m, compute fitted values E[mu|y,theta^m] via 9-pt fast GH
  5. ASSE distribution -> mean, SD, 95% CI

No MCMC needed.  Total runtime: ~25 seconds.

Key scientific question:
  How much does theta-uncertainty inflate the ASSE?
  Plugin (theta fixed) vs Bayesian (theta integrated out).
"""

import sys, os
import numpy as np
from scipy.optimize import minimize
import warnings; warnings.filterwarnings('ignore')

# ── Import Laplace GH loglik from rfl_sev_inla.py ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rfl_sev_inla import loglik as _laplace_loglik_fn

EULER_GAMMA = 0.5772156649015328

# ── Data ─────────────────────────────────────────────────────────
_raw = {
    0.675: [102.95, 280.32, 339.83, 366.9, 485.62, 658.96, 896.33,
            1241.76, 1250.2, 1329.78, 1399.83, 1459.14, 3249.82, 11748.1, 11748.1],
    0.75:  [6.71, 9.93, 12.6, 15.58, 16.19, 17.28, 18.62,
            20.3, 24.9, 26.26, 27.94, 36.35, 48.42, 50.09, 67.34],
    0.825: [1.246, 1.258, 1.46, 1.492, 2.4, 2.41, 2.59,
            2.903, 3.33, 3.59, 3.847, 4.11, 4.82, 5.56, 5.598],
    0.9:   [0.201, 0.216, 0.226, 0.252, 0.257, 0.295, 0.311,
            0.342, 0.356, 0.451, 0.457, 0.509, 0.54, 0.68, 1.129],
    0.95:  [0.037, 0.072, 0.074, 0.076, 0.083, 0.085, 0.105,
            0.109, 0.12, 0.123, 0.143, 0.203, 0.206, 0.217, 0.257],
}
S_r, Y_r = [], []
for s, ys in _raw.items():
    for y in ys: S_r.append(s); Y_r.append(np.log(y))
S_r = np.array(S_r); Y_r = np.array(Y_r)
cens_r = np.zeros(len(Y_r), bool)
n = len(Y_r)

# ── SEV / GH helpers for fast fitted-value computation ───────────
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)
_LOG_SQRT2PI = 0.5 * np.log(2 * np.pi)

def _sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def _fitted_i_fast(yi, si, b0, b1, sig, mu_d, sig_d):
    """E[mu(S,Delta)|y, theta]  via 9-pt GH in z=log(Delta) space."""
    z_j = mu_d + sig_d * _GHX
    d_j = np.exp(z_j)
    valid = d_j < si - 1e-8
    if not valid.any():
        return b0 + b1 * np.log(max(si - np.exp(mu_d), 1e-8)) - sig * EULER_GAMMA
    d_v = d_j[valid]; w_v = _GHW[valid]
    mu_v = b0 + b1 * np.log(si - d_v)
    lf_v = _sev_logpdf(yi, mu_v, sig)
    log_t = lf_v + np.log(np.maximum(w_v, 1e-300))
    lse = log_t.max()
    unnorm = np.exp(log_t - lse)
    wts = unnorm / max(unnorm.sum(), 1e-300)
    return float(np.dot(wts, mu_v)) - sig * EULER_GAMMA

def compute_asse_fast(theta):
    """ASSE using fast GH fitted values (consistent with rfl_sev_inla_mix.py)."""
    b0, b1, lsig, mu_d, lsd = theta
    sig = np.exp(lsig); sig_d = np.exp(lsd)
    obs_mask = ~cens_r
    ae = [abs(_fitted_i_fast(Y_r[i], S_r[i], b0, b1, sig, mu_d, sig_d) - Y_r[i])
          for i in range(n) if obs_mask[i]]
    return float(np.sum(ae))

# ── Laplace GH loglik (wrapper for rfl_sev_inla.py) ───────────────
def loglik_laplace(theta):
    """Total log-likelihood using the published Laplace GH method."""
    return float(_laplace_loglik_fn(theta, Y_r, S_r, cens_r))

# ── Numerical Hessian of loglik (published MLE) ───────────────────
def compute_hessian_laplace(theta, eps=5e-4):
    """
    Numerical Hessian of loglik at theta.
    H_obs = -d^2 logL/dtheta^2  (positive semi-definite at MLE).
    O(p^2) evaluations of the Laplace GH loglik.
    """
    p = len(theta)
    H = np.zeros((p, p))
    f0 = loglik_laplace(theta)
    # Diagonal  (central differences)
    for i in range(p):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        H[i, i] = -(loglik_laplace(tp) - 2*f0 + loglik_laplace(tm)) / eps**2
    # Off-diagonal  (4-point formula)
    for i in range(p):
        for j in range(i+1, p):
            tpp = theta.copy(); tpp[i] += eps; tpp[j] += eps
            tmm = theta.copy(); tmm[i] -= eps; tmm[j] -= eps
            tpm = theta.copy(); tpm[i] += eps; tpm[j] -= eps
            tmp_ = theta.copy(); tmp_[i] -= eps; tmp_[j] += eps
            H[i, j] = -(loglik_laplace(tpp) - loglik_laplace(tpm)
                        - loglik_laplace(tmp_) + loglik_laplace(tmm)) / (4*eps**2)
            H[j, i] = H[i, j]
    return H

# ── Parameter names ───────────────────────────────────────────────
NAT_NAMES = ['beta_0', 'beta_1', 'sigma', 'mu_d', 'sigma_d']

# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import time
    np.random.seed(42)

    print("=" * 70)
    print(f"Direction B: Posterior predictive ASSE  (SEV+INLA,  n={n})")
    print("  p(theta|y) via Laplace approximation  N(theta_hat, H_obs^{-1})")
    print("=" * 70)

    # ── Step 1: Published MLE (Laplace GH) ───────────────────────
    # From rfl_sev_inla.py optimize()  (readme.md values)
    THETA_HAT = np.array([-9.3700, -8.5340, np.log(0.1900), -0.6440, np.log(0.0360)])
    ll_hat = loglik_laplace(THETA_HAT)
    asse_plugin = compute_asse_fast(THETA_HAT)
    aic_hat  = -2*ll_hat + 2*5
    print(f"\n[Step 1] Published MLE (Laplace GH from rfl_sev_inla.py):")
    print(f"  b0={THETA_HAT[0]:.4f}  b1={THETA_HAT[1]:.4f}  "
          f"sig={np.exp(THETA_HAT[2]):.4f}  "
          f"mu_d={THETA_HAT[3]:.4f}  sig_d={np.exp(THETA_HAT[4]):.4f}")
    print(f"  ll={ll_hat:.4f}  AIC={aic_hat:.3f}  ASSE_plugin(fast GH)={asse_plugin:.4f}")

    # ── Step 2: Numerical Hessian ─────────────────────────────────
    print(f"\n[Step 2] Numerical Hessian of Laplace GH loglik ...")
    t0 = time.time()
    H_obs = compute_hessian_laplace(THETA_HAT)
    t1 = time.time()
    print(f"  Hessian computed in {t1-t0:.1f}s")

    # Posterior covariance  =  H_obs^{-1}
    try:
        Cov = np.linalg.inv(H_obs)
        eigvals = np.linalg.eigvalsh(Cov)
        if not all(e > 0 for e in eigvals):
            # Near-singular: regularize with ridge
            lam = abs(min(eigvals)) + 1e-6
            Cov = np.linalg.inv(H_obs + lam * np.eye(5))
            print(f"  Note: Hessian near-singular, ridge lam={lam:.2e} applied")
    except np.linalg.LinAlgError:
        Cov = np.diag(1.0 / np.abs(np.diag(H_obs)))
        print("  Warning: Hessian singular, using diagonal approximation")

    se = np.sqrt(np.diag(Cov))
    nat_se = [se[0], se[1], np.exp(THETA_HAT[2])*se[2],
              se[3], np.exp(THETA_HAT[4])*se[4]]    # delta method
    print(f"\n  Posterior SE (Laplace approx):  [from H_obs^{{-1}}]")
    mle_nat = [THETA_HAT[0], THETA_HAT[1], np.exp(THETA_HAT[2]),
               THETA_HAT[3], np.exp(THETA_HAT[4])]
    for nm, mle_v, se_v in zip(NAT_NAMES, mle_nat, nat_se):
        print(f"    {nm:<10} = {mle_v:8.4f}  SE={se_v:.5f}  "
              f"95% CI = [{mle_v - 1.96*se_v:.4f}, {mle_v + 1.96*se_v:.4f}]")

    # ── Step 3: Sample from posterior Gaussian ────────────────────
    M = 2000
    print(f"\n[Step 3] Sampling {M} theta ~ N(theta_hat, H_obs^{{-1}}) ...")
    samples = np.random.multivariate_normal(THETA_HAT, Cov, size=M)
    # Report posterior in natural scale
    nat_samples = samples.copy()
    nat_samples[:, 2] = np.exp(samples[:, 2])   # log_sig -> sig
    nat_samples[:, 4] = np.exp(samples[:, 4])   # log_sig_d -> sig_d
    print(f"\n  Posterior statistics (Gaussian samples):")
    print(f"  {'Param':<10}  {'Post.Mean':>10}  {'Post.SD':>8}  "
          f"{'2.5%':>8}  {'97.5%':>8}  {'MLE':>8}")
    for j, nm in enumerate(NAT_NAMES):
        col = nat_samples[:, j]
        print(f"  {nm:<10}  {col.mean():>10.4f}  {col.std():>8.5f}  "
              f"{np.percentile(col, 2.5):>8.4f}  {np.percentile(col, 97.5):>8.4f}  "
              f"{mle_nat[j]:>8.4f}")

    # ── Step 4: ASSE distribution over posterior ──────────────────
    print(f"\n[Step 4] Computing ASSE for each of {M} posterior samples ...")
    t0 = time.time()
    asse_samples = np.array([compute_asse_fast(samples[m]) for m in range(M)])
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")

    asse_plugin_ref = 5.76   # from readme.md (Laplace GH fitted values)
    print(f"\n  Reference (readme.md):  ASSE_plugin(Laplace GH) = {asse_plugin_ref:.4f}")
    print(f"  Plugin at theta_hat (fast GH):               = {asse_plugin:.4f}")
    print(f"  ASSE_bayes  E_theta[ASSE]                    = {asse_samples.mean():.4f}")
    print(f"  ASSE_bayes  SD                               = {asse_samples.std():.4f}")
    print(f"  ASSE_bayes  95% CI                           = "
          f"[{np.percentile(asse_samples, 2.5):.4f}, "
          f"{np.percentile(asse_samples, 97.5):.4f}]")
    print(f"  ASSE_bayes  median                           = "
          f"{np.percentile(asse_samples, 50):.4f}")

    inflation = asse_samples.mean() / asse_plugin if asse_plugin > 0 else np.nan
    print(f"\n  theta-uncertainty inflation factor: {inflation:.4f}  "
          f"(+{(inflation-1)*100:.2f}%)")

    # ── Step 5: Per-parameter sensitivity ─────────────────────────
    print(f"\n[Step 5] ASSE sensitivity to each parameter (one-at-a-time):")
    print(f"  {'Param':<10}  {'ASSE(MLE+1SE)':>14}  {'ASSE(MLE-1SE)':>14}  "
          f"{'delta':>8}")
    for j, nm in enumerate(NAT_NAMES):
        th_plus = THETA_HAT.copy()
        th_minus = THETA_HAT.copy()
        th_plus[j]  += se[j]
        th_minus[j] -= se[j]
        ae_plus  = compute_asse_fast(th_plus)
        ae_minus = compute_asse_fast(th_minus)
        print(f"  {nm:<10}  {ae_plus:>14.4f}  {ae_minus:>14.4f}  "
              f"{abs(ae_plus-ae_minus)/2:>+8.4f}")

    # ── Step 6: LOO contrast ──────────────────────────────────────
    print(f"\n[Step 6] Comparison with LOO results (from rfl_loo.py):")
    print(f"  Prior-LOO ASSE (all methods)              ~ 40")
    print(f"  Posterior Sharpening Bonus (SEV+INLA)     ~ 34.6")
    print(f"  ASSE_plugin  (in-sample, theta fixed)     = {asse_plugin_ref:.2f}")
    print(f"  ASSE_bayes   (in-sample, theta integrated)= {asse_samples.mean():.2f}")
    print(f"")
    print(f"  LOO insight: integrating out theta does NOT help LOO prediction.")
    print(f"  The gap (40 -> {asse_plugin_ref:.2f}) is due to per-obs Δᵢ conditioning,")
    print(f"  not theta estimation uncertainty.")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY  Direction B: Posterior predictive ASSE")
    print(f"{'='*70}")
    print(f"  Posterior (Laplace approx):    N(theta_hat, H_obs^{{-1}})")
    print(f"  M samples:                     {M}")
    print(f"  ASSE_plugin (theta fixed):     {asse_plugin_ref:.4f}  [Laplace GH]")
    print(f"  ASSE_bayes  (E_theta[ASSE]):   {asse_samples.mean():.4f} "
          f"+/- {asse_samples.std():.4f}")
    print(f"  95% credible interval:         "
          f"[{np.percentile(asse_samples,2.5):.4f}, "
          f"{np.percentile(asse_samples,97.5):.4f}]")
    print(f"  Inflation from theta-SE:       +{(inflation-1)*100:.2f}%")
    print(f"")
    print(f"  Scientific conclusion:")
    print(f"  Integrating out theta adds only ~{(inflation-1)*100:.1f}% to ASSE.")
    print(f"  The dominant uncertainty source is the latent Δᵢ (per-obs),")
    print(f"  not the global parameter vector theta.  LOO ASSE ~ 40 >> {asse_plugin_ref}.")
