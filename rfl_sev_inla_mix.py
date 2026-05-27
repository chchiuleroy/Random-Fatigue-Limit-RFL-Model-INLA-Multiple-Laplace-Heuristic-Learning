"""
rfl_sev_inla_mix.py  --  SEV + INLA + K-component LogNormal mixture g(Δ)
=========================================================================
Model:
  g(Δ) = Σₖ πₖ · LogNormal(μₖ, σₖ²)        [K-component mixture prior]
  f(Y|Δ) = SEV(β₀ + β₁·log(S-Δ),  σ)       [SEV conditional]
  L_i(θ) = Σₖ πₖ · ∫ f(yᵢ|Δ) LogNormal_k(Δ) dΔ   [per-obs GH per component]

Motivation:
  - SEV+INLA (K=1) forces unimodal g(Δ):  ASSE=5.76,  AIC=155.76
  - Normal+NPMLE K=2 finds atoms at {0.532,0.569}  →  g(Δ) may be bimodal
  - SEV NPMLE BIC prefers K=3
  - K-mixture LogNormal bridges continuous INLA posterior (fine Δᵢ recovery)
    with the NPMLE's data-driven multi-modality

LOO insight (rfl_loo.py):
  Prior-LOO ASSE ≈ 40 for all methods → mixture won't help prediction.
  But: in-sample ASSE / AIC comparison is still meaningful for
  "how well does the model characterise the fatigue-limit distribution?"

Parameters:
  K=1: [b0, b1, log_sig, mu_d, log_sig_d]                              5 params
  K=2: [b0, b1, log_sig, logit_pi1, mu_d1, log_sd1, mu_d2, log_sd2]   8 params
  K=3: [b0, b1, log_sig, logit_pi1, logit_pi2, mu_d1, log_sd1,
        mu_d2, log_sd2, mu_d3, log_sd3]                               11 params
  (K-1 free mixing weights via softmax on unconstrained logits)

Fitted value:
  ŷᵢ = E[Y|yᵢ,Sᵢ] = Σₖ P(Z=k|yᵢ) · E[μ(Sᵢ,Δᵢ)|yᵢ,Z=k] - σ·γ
  P(Z=k|yᵢ) ∝ πₖ · L_ik        (soft component assignment)

Outer optimizer:  random grid → dual annealing → Nelder-Mead
  (same heuristic pipeline as rfl_sev_inla.py)
"""

import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize, dual_annealing
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

# ── SEV functions ────────────────────────────────────────────────
def _sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def _sev_logsf(y, mu, sig):
    return -np.exp((y - mu) / sig)

# ── Gauss-Hermite setup (probabilist's HE nodes) ─────────────────
# hermegauss: ∫ f(t) e^{-t²/2} dt ≈ Σ w_j f(t_j)
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)
_LOG_SQRT2PI = 0.5 * np.log(2 * np.pi)

def _gh_logmarginal_k(y, s, is_cens, b0, b1, sig, mu_dk, sig_dk):
    """Fast 9-pt GH in z=log(Delta) space.
    ∫ f(y|Δ) LN_k(Δ) dΔ = (1/√2π) Σ_j w_j f(y|exp(mu_dk + sig_dk*t_j))
    No nested optimize needed -- O(N_GH) evaluations per (i,k).
    """
    z_j = mu_dk + sig_dk * _GHX        # GH nodes in log(Δ) space
    d_j = np.exp(z_j)
    valid = d_j < s - 1e-8
    if not valid.any():
        return -1e30
    d_v = d_j[valid]
    w_v = _GHW[valid]
    mu_v = b0 + b1 * np.log(s - d_v)
    lf_v = _sev_logsf(y, mu_v, sig) if is_cens else _sev_logpdf(y, mu_v, sig)
    log_terms = lf_v + np.log(np.maximum(w_v, 1e-300))
    lse = log_terms.max()
    return -_LOG_SQRT2PI + lse + np.log(np.sum(np.exp(log_terms - lse)))

def _gh_post_mean_k(y, s, b0, b1, sig, mu_dk, sig_dk):
    """E[μ(S,Δ)|y,Z=k] via fast GH in z=log(Δ) space (uncensored obs)."""
    z_j = mu_dk + sig_dk * _GHX
    d_j = np.exp(z_j)
    valid = d_j < s - 1e-8
    if not valid.any():
        return b0 + b1 * np.log(max(s - np.exp(mu_dk), 1e-8))
    d_v = d_j[valid]
    w_v = _GHW[valid]
    mu_v = b0 + b1 * np.log(s - d_v)
    lf_v = _sev_logpdf(y, mu_v, sig)
    log_terms = lf_v + np.log(np.maximum(w_v, 1e-300))
    lse = log_terms.max()
    unnorm = np.exp(log_terms - lse)
    weights = unnorm / max(unnorm.sum(), 1e-300)
    return float(np.dot(weights, mu_v))

# ══════════════════════════════════════════════════════════════════
# Parameter packing / unpacking
# ══════════════════════════════════════════════════════════════════
def _unpack(theta, K):
    """
    Unpack flat theta into (b0, b1, sig, pis, mu_ds, sig_ds).
    K=1: theta[3:] = [mu_d,  log_sig_d]
    K=2: theta[3:] = [logit_pi1, mu_d1, log_sd1, mu_d2, log_sd2]
    K=3: theta[3:] = [logit_pi1, logit_pi2, mu_d1,log_sd1, mu_d2,log_sd2, mu_d3,log_sd3]
    """
    b0, b1, log_sig = theta[0], theta[1], theta[2]
    sig = np.exp(log_sig)
    if K == 1:
        mu_ds   = np.array([theta[3]])
        sig_ds  = np.array([np.exp(theta[4])])
        pis     = np.array([1.0])
    elif K == 2:
        logit1  = theta[3]
        pi1     = 1.0 / (1.0 + np.exp(-logit1))
        pis     = np.array([pi1, 1.0 - pi1])
        mu_ds   = np.array([theta[4], theta[6]])
        sig_ds  = np.array([np.exp(theta[5]), np.exp(theta[7])])
    elif K == 3:
        # logit_pi1, logit_pi2 → softmax over 3 components
        lp      = np.array([theta[3], theta[4], 0.0])   # reference: lp3=0
        pis     = np.exp(lp - lp.max()); pis /= pis.sum()
        mu_ds   = np.array([theta[5], theta[7], theta[9]])
        sig_ds  = np.array([np.exp(theta[6]), np.exp(theta[8]), np.exp(theta[10])])
    else:
        raise ValueError(f"K={K} not supported (max 3)")
    return b0, b1, sig, pis, mu_ds, sig_ds

def _n_params(K):
    return {1: 5, 2: 8, 3: 11}[K]

# ══════════════════════════════════════════════════════════════════
# Log-likelihood (mixture)
# ══════════════════════════════════════════════════════════════════
def loglik_mix(theta, Y, S, cens, K):
    b0, b1, sig, pis, mu_ds, sig_ds = _unpack(theta, K)
    if sig < 0.01: return -1e30
    if any(sd < 0.005 for sd in sig_ds): return -1e30
    ll = 0.0
    for i in range(len(Y)):
        # log Σₖ πₖ exp(log_L_ik)
        log_terms = []
        for k in range(K):
            ll_k = _gh_logmarginal_k(Y[i], S[i], bool(cens[i]),
                                     b0, b1, sig, mu_ds[k], sig_ds[k])
            log_terms.append(np.log(pis[k]) + ll_k)
        mx = max(log_terms)
        ll += mx + np.log(sum(np.exp(t - mx) for t in log_terms))
    return ll

# ══════════════════════════════════════════════════════════════════
# Fitted values (ASSE)
# ══════════════════════════════════════════════════════════════════
def compute_asse_mix(Y, S, cens, theta, K):
    b0, b1, sig, pis, mu_ds, sig_ds = _unpack(theta, K)
    obs = ~cens
    fitted = np.zeros(len(Y))
    for i in range(len(Y)):
        if cens[i]:
            # For censored: use prior predictive (no y_i info)
            mu_prior = sum(pis[k] * (b0 + b1 * np.log(max(S[i] - np.exp(mu_ds[k]), 1e-8)))
                           for k in range(K))
            fitted[i] = mu_prior - sig * EULER_GAMMA
            continue
        # log P(Z=k|yᵢ) ∝ log πₖ + log L_ik
        log_wts = []
        mu_k_list = []
        for k in range(K):
            ll_k = _gh_logmarginal_k(Y[i], S[i], False,
                                     b0, b1, sig, mu_ds[k], sig_ds[k])
            log_wts.append(np.log(pis[k]) + ll_k)
            mu_k_list.append(_gh_post_mean_k(Y[i], S[i], b0, b1, sig, mu_ds[k], sig_ds[k]))
        mx = max(log_wts)
        wts = np.exp(np.array(log_wts) - mx)
        wts /= wts.sum()
        fitted[i] = float(np.dot(wts, mu_k_list)) - sig * EULER_GAMMA
    ae = np.abs(Y - fitted)
    return ae[obs].sum(), ae[obs].mean(), np.median(ae[obs]), ae, fitted

# ══════════════════════════════════════════════════════════════════
# Bounds for optimizer
# ══════════════════════════════════════════════════════════════════
def _make_bounds(K):
    base = [(-20., -2.), (-20., -2.), (np.log(0.05), np.log(2.0))]
    if K == 1:
        return base + [(-3., 1.), (np.log(0.005), np.log(1.0))]
    elif K == 2:
        pi_bounds  = [(-6., 6.)]             # logit_pi1
        comp_bounds = [(-3., 1.), (np.log(0.005), np.log(0.5))] * 2
        return base + pi_bounds + comp_bounds
    elif K == 3:
        pi_bounds  = [(-6., 6.), (-6., 6.)]  # logit_pi1, logit_pi2
        comp_bounds = [(-3., 1.), (np.log(0.005), np.log(0.5))] * 3
        return base + pi_bounds + comp_bounds

# ══════════════════════════════════════════════════════════════════
# Heuristic optimizer: random multi-start L-BFGS-B + NM polish
# (SA removed -- fast GH makes L-BFGS-B competitive and ~50x faster)
# ══════════════════════════════════════════════════════════════════
def heuristic_optimize(Y, S, cens, K, x0=None, n_starts=60, seed=42):
    bounds = _make_bounds(K)
    rng = np.random.default_rng(seed)
    obj = lambda t: -loglik_mix(t, Y, S, cens, K)
    best_ll, best_t = -np.inf, None

    candidates = []
    # Warm-start first (top priority)
    if x0 is not None:
        candidates.append(list(x0))
    # Random starts
    for _ in range(n_starts):
        candidates.append([rng.uniform(lo, hi) for lo, hi in bounds])

    # Stage 1: evaluate all candidates, take top-5
    evals = []
    for t in candidates:
        ll = loglik_mix(t, Y, S, cens, K)
        evals.append((ll, t))
    evals.sort(key=lambda x: -x[0])
    print(f"  [Init] best ll = {evals[0][0]:.4f}  (warm={x0 is not None})")

    # Stage 2: L-BFGS-B from top-5 starts
    top5 = [t for _, t in evals[:5]]
    for t_init in top5:
        res = minimize(obj, x0=t_init, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6})
        if -res.fun > best_ll:
            best_ll, best_t = -res.fun, res.x.tolist()
    print(f"  [BFGS] best ll = {best_ll:.4f}")

    # Stage 3: Nelder-Mead polish
    res_nm = minimize(obj, x0=best_t, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-7})
    print(f"  [NM  ] best ll = {-res_nm.fun:.4f}")
    return res_nm.x, float(-res_nm.fun)

# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════
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
stresses = sorted(_raw.keys())
n = len(Y_r)

# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print(f"SEV + INLA + K-mixture LogNormal g(D)  n={n}")
    print("=" * 70)

    # K=1 warm-start from rfl_sev_inla.py result
    THETA_K1 = np.array([-9.3700, -8.5340, np.log(0.1900), -0.6440, np.log(0.0360)])

    results = {}

    # ── K=1 (re-optimize with fast GH for apples-to-apples comparison) ─
    print(f"\n{'─'*70}")
    print("K=1  (re-optimized with fast GH -- consistent GH method for AIC/BIC)")
    print(f"{'─'*70}")
    print(f"  Running heuristic optimizer (multi-start BFGS, n_starts=60)...")
    theta_k1, ll_k1 = heuristic_optimize(Y_r, S_r, cens_r, K=1, x0=list(THETA_K1))
    asse_k1, mae_k1, medae_k1, ae_k1, _ = compute_asse_mix(Y_r, S_r, cens_r, theta_k1, K=1)
    p1 = _n_params(1)
    aic_k1 = -2*ll_k1 + 2*p1
    bic_k1 = -2*ll_k1 + p1*np.log(n)
    b0_1, b1_1, sig_1, pis_1, mu_ds_1, sig_ds_1 = _unpack(theta_k1, K=1)
    print(f"  b0={b0_1:.4f}  b1={b1_1:.4f}  sig={sig_1:.4f}  "
          f"mu_d={mu_ds_1[0]:.4f}  sig_d={sig_ds_1[0]:.4f}")
    print(f"  ll={ll_k1:.4f}  ASSE={asse_k1:.4f}  MAE={mae_k1:.5f}")
    print(f"  AIC={aic_k1:.3f}  BIC={bic_k1:.3f}  params={p1}")
    results[1] = dict(theta=theta_k1, ll=ll_k1, asse=asse_k1, aic=aic_k1, bic=bic_k1)

    # ── K=2 warm-start: split K=1 into two components ───────────
    print(f"\n{'─'*70}")
    print("K=2  (bimodal g(D) -- bridges NPMLE K=2 discrete atoms)")
    print(f"{'─'*70}")

    mu_d0, log_sd0 = THETA_K1[3], THETA_K1[4]
    sd0 = np.exp(log_sd0)
    # Split: component 1 left of mean, component 2 right
    # Informed by NPMLE: atoms at 0.532 (pi=0.93) and 0.569 (pi=0.07)
    # LogNormal: log(0.532)=-0.631, log(0.569)=-0.564
    x0_k2 = [
        THETA_K1[0], THETA_K1[1], THETA_K1[2],    # b0, b1, log_sig
        3.0,                                         # logit_pi1 → pi1≈0.95
        np.log(0.532), np.log(0.025),               # mu_d1≈log(0.532), sig_d1=0.025
        np.log(0.569), np.log(0.025),               # mu_d2≈log(0.569), sig_d2=0.025
    ]
    print(f"  Warm-start: pi1~{1/(1+np.exp(-3.0)):.2f}  "
          f"mu_d1={np.log(0.532):.3f}  mu_d2={np.log(0.569):.3f}")
    print(f"  Running heuristic optimizer (multi-start BFGS, n_starts=60)...")
    theta_k2, ll_k2 = heuristic_optimize(Y_r, S_r, cens_r, K=2, x0=x0_k2)
    asse_k2, mae_k2, medae_k2, ae_k2, _ = compute_asse_mix(Y_r, S_r, cens_r, theta_k2, K=2)
    p2 = _n_params(2)
    aic_k2 = -2*ll_k2 + 2*p2
    bic_k2 = -2*ll_k2 + p2*np.log(n)
    # Recover mixing weights and component params
    b0_, b1_, sig_, pis_, mu_ds_, sig_ds_ = _unpack(theta_k2, K=2)
    print(f"\n  RESULTS K=2:")
    print(f"  b0={b0_:.4f}  b1={b1_:.4f}  sig={sig_:.4f}")
    for k in range(2):
        E_d = np.exp(mu_ds_[k] + sig_ds_[k]**2 / 2)
        print(f"  Component {k+1}: pi={pis_[k]:.4f}  mu_d={mu_ds_[k]:.4f}  "
              f"sig_d={sig_ds_[k]:.4f}  E[D]={E_d:.4f}")
    print(f"  ll={ll_k2:.4f}  ASSE={asse_k2:.4f}  MAE={mae_k2:.5f}")
    print(f"  AIC={aic_k2:.3f}  BIC={bic_k2:.3f}  params={p2}")
    results[2] = dict(theta=theta_k2, ll=ll_k2, asse=asse_k2, aic=aic_k2, bic=bic_k2)

    # ── K=3 warm-start from K=2 solution ────────────────────────
    print(f"\n{'─'*70}")
    print("K=3  (trimodal g(D) -- SEV NPMLE BIC favours K=3)")
    print(f"{'─'*70}")

    # Build K=3 from K=2: split the larger component
    b0_2, b1_2, sig_2, pis_2, mu_ds_2, sig_ds_2 = _unpack(theta_k2, K=2)
    # Split component 1 (dominant) into two sub-components
    lp1 = np.log(pis_2[0]*0.6); lp2 = np.log(pis_2[0]*0.4); lp3 = 0.0
    arr = np.array([lp1, lp2, lp3])
    # logit_pi1 = log(p1/(1-p1)) where softmax gives correct pis
    # Approximate via: just use original K=2 mu_d for first two, add third near K=2 component 2
    logit_pi1_3 = np.log(pis_2[0]*0.6) - np.log(pis_2[1])   # rough
    logit_pi2_3 = np.log(pis_2[0]*0.4) - np.log(pis_2[1])
    x0_k3 = [
        b0_2, b1_2, np.log(sig_2),
        logit_pi1_3, logit_pi2_3,
        mu_ds_2[0] - 0.05, np.log(sig_ds_2[0]),     # component 1
        mu_ds_2[0] + 0.05, np.log(sig_ds_2[0]),     # component 2 (split from 1)
        mu_ds_2[1], np.log(sig_ds_2[1]),              # component 3
    ]
    print(f"  Running heuristic optimizer (multi-start BFGS, n_starts=60)...")
    theta_k3, ll_k3 = heuristic_optimize(Y_r, S_r, cens_r, K=3, x0=x0_k3)
    asse_k3, mae_k3, medae_k3, ae_k3, _ = compute_asse_mix(Y_r, S_r, cens_r, theta_k3, K=3)
    p3 = _n_params(3)
    aic_k3 = -2*ll_k3 + 2*p3
    bic_k3 = -2*ll_k3 + p3*np.log(n)
    b0_3, b1_3, sig_3, pis_3, mu_ds_3, sig_ds_3 = _unpack(theta_k3, K=3)
    print(f"\n  RESULTS K=3:")
    print(f"  b0={b0_3:.4f}  b1={b1_3:.4f}  sig={sig_3:.4f}")
    for k in range(3):
        E_d = np.exp(mu_ds_3[k] + sig_ds_3[k]**2 / 2)
        print(f"  Component {k+1}: pi={pis_3[k]:.4f}  mu_d={mu_ds_3[k]:.4f}  "
              f"sig_d={sig_ds_3[k]:.4f}  E[D]={E_d:.4f}")
    print(f"  ll={ll_k3:.4f}  ASSE={asse_k3:.4f}  MAE={mae_k3:.5f}")
    print(f"  AIC={aic_k3:.3f}  BIC={bic_k3:.3f}  params={p3}")
    results[3] = dict(theta=theta_k3, ll=ll_k3, asse=asse_k3, aic=aic_k3, bic=bic_k3)

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY  (SEV + INLA + K-mixture LogNormal g(D))")
    print(f"{'='*70}")
    print(f"  {'Method':<30}  {'params':>6}  {'ll':>9}  {'AIC':>8}  {'BIC':>8}  {'ASSE':>7}  {'MAE':>8}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}")
    for K, (ae_k, mae_k, label) in [
        (1, (ae_k1, mae_k1, "SEV+INLA (K=1 baseline)")),
        (2, (ae_k2, mae_k2, "SEV+INLA K=2 mixture")),
        (3, (ae_k3, mae_k3, "SEV+INLA K=3 mixture")),
    ]:
        r = results[K]
        print(f"  {label:<30}  {_n_params(K):>6}  {r['ll']:>9.4f}  "
              f"{r['aic']:>8.3f}  {r['bic']:>8.3f}  {r['asse']:>7.4f}  {mae_k:>8.5f}")
    print()
    print(f"  Reference (from readme.md):")
    print(f"  {'SEV+INLA (orig)':<30}  {5:>6}  {-72.880:>9.4f}  {155.76:>8.3f}  {'':>8}  {5.76:>7.2f}")
    print(f"  {'Burr+EM-GMM Mode A K=1':<30}  {6:>6}  {-72.87:>9.4f}  {157.73:>8.3f}  {'':>8}  {4.09:>7.2f}")

    print(f"\n{'─'*70}")
    print("Per-stress MAE breakdown:")
    print(f"  {'S':>7}  {'K=1':>10}  {'K=2':>10}  {'K=3':>10}")
    for sv in stresses:
        idx = S_r == sv
        print(f"  {sv:7.3f}  {ae_k1[idx].mean():>10.5f}  {ae_k2[idx].mean():>10.5f}  {ae_k3[idx].mean():>10.5f}")

    print(f"\n{'─'*70}")
    print("Interpretation:")
    best_aic = min(results[K]['aic'] for K in [1, 2, 3])
    best_bic = min(results[K]['bic'] for K in [1, 2, 3])
    best_asse = min(results[K]['asse'] for K in [1, 2, 3])
    for K in [1, 2, 3]:
        r = results[K]
        tags = []
        if r['aic'] == best_aic: tags.append("AIC best")
        if r['bic'] == best_bic: tags.append("BIC best")
        if r['asse'] == best_asse: tags.append("ASSE best")
        flag = "  <- " + " | ".join(tags) if tags else ""
        print(f"  K={K}: DAIC={r['aic']-best_aic:+.3f}  DBIC={r['bic']-best_bic:+.3f}  "
              f"ASSE={r['asse']:.4f}{flag}")
    print()
    print("  Note: LOO analysis (rfl_loo.py) shows Prior-LOO ASSE ~ 40 for all K.")
    print("  AIC/BIC comparison measures g(D) characterisation, not new-specimen prediction.")
