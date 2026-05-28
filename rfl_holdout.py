"""
rfl_holdout.py  —  Stress-group holdout validation for SEV+INLA
================================================================
Train : S in {0.675, 0.750, 0.825, 0.900}   n=60
Test  : S = 0.950                             n=15  (never seen during fit)

Step 1 — Re-fit SEV+INLA on n=60 training set.
Step 2 — Predict test set using re-fitted theta (no y_i conditioning):
  A. Direct marginal CDF:
       P(Y<=t|S) = E_{g(Delta)}[F_SEV(t; mu(S,Delta), sig)]  via 9-pt GH
       -> Q_q(Y|S) via brentq
  B. Gaussian mixture approximation (closed-form CDF):
       At each GH node: SEV(mu_j, sig) ~ N(mu_j - sig*gamma, sig^2*pi^2/6)
       -> F_mix(t) = sum_j w_j * Phi((t - yhat_j)/sigma_mix)
  C. Delta-quantile: mu(S, Delta^(q)) - sig*gamma  for q in {0.10..0.90}

Reference: in-sample posterior (uses y_i, double-dipping) for comparison.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm as scipy_norm

from rfl_sev_inla import (
    _GHX, _GHW,
    _log_h, _posterior_mean_mu,
    loglik, _BOUNDS,
    EULER_GAMMA,
)

_SQRT2PI = np.sqrt(2 * np.pi)
SEV_VAR_FACTOR = np.pi**2 / 6   # Var(SEV) = sig^2 * pi^2/6

# ── Known MLE from full n=75 fit (warm-start only) ───────────────────────────
THETA_N75 = np.array([-9.3700, -8.5340, np.log(0.1900), -0.6440, np.log(0.0360)])

# ── Data ─────────────────────────────────────────────────────────────────────
_raw = {
    0.675: [102.95,280.32,339.83,366.9,485.62,658.96,896.33,
            1241.76,1250.2,1329.78,1399.83,1459.14,3249.82,11748.1,11748.1],
    0.75:  [6.71,9.93,12.6,15.58,16.19,17.28,18.62,
            20.3,24.9,26.26,27.94,36.35,48.42,50.09,67.34],
    0.825: [1.246,1.258,1.46,1.492,2.4,2.41,2.59,
            2.903,3.33,3.59,3.847,4.11,4.82,5.56,5.598],
    0.9:   [0.201,0.216,0.226,0.252,0.257,0.295,0.311,
            0.342,0.356,0.451,0.457,0.509,0.54,0.68,1.129],
    0.95:  [0.037,0.072,0.074,0.076,0.083,0.085,0.105,
            0.109,0.12,0.123,0.143,0.203,0.206,0.217,0.257],
}
S_all, Y_all = [], []
for s, ys in _raw.items():
    for y in ys:
        S_all.append(s); Y_all.append(np.log(y))
S_all = np.array(S_all); Y_all = np.array(Y_all)

TEST_S = 0.95
Y_test  = Y_all[S_all == TEST_S]     # n=15
n_test  = len(Y_test)
n_train = int((S_all != TEST_S).sum())  # 60

# ── Unpack theta ──────────────────────────────────────────────────────────────
def unpack(theta):
    b0, b1, log_sig, mu_d, log_sig_d = theta
    return b0, b1, np.exp(log_sig), mu_d, np.exp(log_sig_d)

# ── GH nodes for given (mu_d, sig_d) ─────────────────────────────────────────
def gh_nodes(mu_d, sig_d, s):
    """Return (delta_j, w_j, mu_j) arrays for valid GH nodes at stress s."""
    z_j  = mu_d + sig_d * _GHX
    d_j  = np.exp(z_j)
    valid = (d_j > 0) & (d_j < s - 1e-8)
    d_v  = d_j[valid]
    w_v  = _GHW[valid] / _SQRT2PI   # normalised probabilist weights
    return d_v, w_v

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH A — Direct marginal CDF via GH
# ══════════════════════════════════════════════════════════════════════════════
def marginal_mean(s, theta):
    """E_{g(Delta)}[mu(S,Delta)] - sig*gamma  (= marginal mean of Y|S)."""
    b0, b1, sig, mu_d, sig_d = unpack(theta)
    d_v, w_v = gh_nodes(mu_d, sig_d, s)
    mu_v = b0 + b1 * np.log(s - d_v)
    return float(np.dot(w_v, mu_v)) - sig * EULER_GAMMA

def marginal_cdf_A(t, s, theta):
    """P(Y <= t | S) via 9-pt GH.  F_SEV(t;mu,sig)=1-exp(-exp((t-mu)/sig))."""
    b0, b1, sig, mu_d, sig_d = unpack(theta)
    d_v, w_v = gh_nodes(mu_d, sig_d, s)
    mu_v  = b0 + b1 * np.log(s - d_v)
    F_sev = 1.0 - np.exp(-np.exp((t - mu_v) / sig))
    return float(np.dot(w_v, F_sev))

def quantile_A(q, s, theta, bracket_width=5.0):
    """Q_q(Y|S) via bisection on marginal CDF (Approach A)."""
    mu_hat = marginal_mean(s, theta)
    _, _, sig, _, _ = unpack(theta)
    lo, hi = mu_hat - bracket_width * sig, mu_hat + bracket_width * sig
    try:
        return brentq(lambda t: marginal_cdf_A(t, s, theta) - q, lo, hi, xtol=1e-6)
    except ValueError:
        return mu_hat

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH B — Gaussian mixture approximation (closed-form)
# ══════════════════════════════════════════════════════════════════════════════
def _gmix_params(s, theta):
    """
    Approx each SEV(mu_j, sig) as N(mu_j - sig*gamma, sig^2 * pi^2/6).
    Returns (yhat_j, sig_mix, w_j) for the 9-component Gaussian mixture.
    """
    b0, b1, sig, mu_d, sig_d = unpack(theta)
    d_v, w_v = gh_nodes(mu_d, sig_d, s)
    mu_v    = b0 + b1 * np.log(s - d_v)
    yhat_v  = mu_v - sig * EULER_GAMMA       # component means
    sig_mix = sig * np.sqrt(SEV_VAR_FACTOR)  # component std (same for all)
    return yhat_v, sig_mix, w_v

def marginal_cdf_B(t, s, theta):
    """CDF of Gaussian mixture: F_mix(t) = sum_j w_j * Phi((t - yhat_j)/sig_mix)."""
    yhat_v, sig_mix, w_v = _gmix_params(s, theta)
    return float(np.dot(w_v, scipy_norm.cdf((t - yhat_v) / sig_mix)))

def quantile_B(q, s, theta):
    """Q_q via bisection on Gaussian mixture CDF (Approach B)."""
    mu_hat = marginal_mean(s, theta)
    _, _, sig, _, _ = unpack(theta)
    sig_mix = sig * np.sqrt(SEV_VAR_FACTOR)
    lo, hi  = mu_hat - 6 * sig_mix, mu_hat + 4 * sig_mix
    try:
        return brentq(lambda t: marginal_cdf_B(t, s, theta) - q, lo, hi, xtol=1e-6)
    except ValueError:
        return mu_hat

# ══════════════════════════════════════════════════════════════════════════════
# DELTA-QUANTILE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict_dq(q, s, theta):
    """yhat(q) = mu(S, Delta^(q)) - sig*gamma  where Delta^(q) = LN quantile."""
    b0, b1, sig, mu_d, sig_d = unpack(theta)
    d_q = np.exp(mu_d + sig_d * scipy_norm.ppf(q))
    if d_q >= s - 1e-8:
        return np.nan
    return b0 + b1 * np.log(s - d_q) - sig * EULER_GAMMA

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# ── Re-fit on training set ────────────────────────────────────────────────────
def fit_train(Y, S, cens, seed=42):
    """Multi-start L-BFGS-B + NM polish on training data."""
    rng = np.random.default_rng(seed)
    obj = lambda t: -loglik(t, Y, S, cens)

    # Warm-start candidates: n=75 MLE + random starts
    candidates = [THETA_N75.tolist()]
    for _ in range(7):
        candidates.append([rng.uniform(lo, hi) for lo, hi in _BOUNDS])

    evals = [(loglik(np.array(t), Y, S, cens), t) for t in candidates]
    evals.sort(key=lambda x: -x[0])
    best_t = np.array(evals[0][1])
    print(f"  Grid best ll = {evals[0][0]:.4f}  (n75 MLE ll = {evals[1][0]:.4f})")

    res_b = minimize(obj, x0=best_t, method='L-BFGS-B', bounds=_BOUNDS,
                     options={'ftol': 1e-10, 'maxiter': 500})
    print(f"  BFGS  ll = {-res_b.fun:.4f}")

    # NM polish — tighter stop to finish faster
    res_n = minimize(obj, x0=res_b.x, method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    print(f"  NM    ll = {-res_n.fun:.4f}")
    return res_n.x, float(-res_n.fun)


if __name__ == '__main__':
    SEP = '=' * 65
    print(SEP)
    print("Stress-group holdout validation  |  SEV + INLA")
    print(f"  Train: S in {{0.675, 0.750, 0.825, 0.900}}  n={n_train}")
    print(f"  Test : S={TEST_S}  n={n_test}  (never seen during fit)")
    print(SEP)

    # ── Step 1: Re-fit on training set ────────────────────────────
    print(f"\n[1] Re-fitting SEV+INLA on n={n_train} training set...")
    S_train = S_all[S_all != TEST_S]; Y_train = Y_all[S_all != TEST_S]
    cens_train = np.zeros(n_train, bool)
    theta_tr, ll_tr = fit_train(Y_train, S_train, cens_train)

    b0, b1, sig, mu_d, sig_d = unpack(theta_tr)
    E_d   = np.exp(mu_d + sig_d**2 / 2)
    b0_75, b1_75, sig_75, mu_d_75, sig_d_75 = unpack(THETA_N75)

    print(f"\n  {'Param':<8} {'n=60 train':>12} {'n=75 full':>12} {'Delta':>9}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*9}")
    for nm, v60, v75 in [('b0', b0, b0_75), ('b1', b1, b1_75),
                          ('sig', sig, sig_75), ('mu_d', mu_d, mu_d_75),
                          ('sig_d', sig_d, sig_d_75)]:
        print(f"  {nm:<8} {v60:>12.5f} {v75:>12.5f} {v60-v75:>+9.5f}")
    print(f"  {'E[Delta]':<8} {E_d:>12.5f} {np.exp(mu_d_75+sig_d_75**2/2):>12.5f}")

    # ── Step 2: Predictions using re-fitted theta_tr ───────────────
    print(f"\n[2] Predictions at S={TEST_S} using re-fitted theta (n=60)")

    # Marginal mean & quantiles
    yhat_mean = marginal_mean(TEST_S, theta_tr)
    print(f"  Marginal mean E[Y|S={TEST_S}] = {yhat_mean:.4f}")
    print(f"  (actual test mean              = {Y_test.mean():.4f})")

    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    print(f"\n  {'q':>5}  {'Approach A':>12}  {'Approach B':>12}  {'Delta-quant':>12}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}")
    dq_vals = {}
    for q in qs:
        qa = quantile_A(q, TEST_S, theta_tr)
        qb = quantile_B(q, TEST_S, theta_tr)
        dq = predict_dq(q, TEST_S, theta_tr)
        dq_vals[q] = dq
        print(f"  {q:5.2f}  {qa:12.4f}  {qb:12.4f}  {dq:12.4f}")

    # ── ASSE comparison ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"[3] ASSE on test set  (n={n_test}, S={TEST_S})")
    print(f"    theta fitted on n={n_train} — completely held-out test")
    print(f"{'─'*65}")
    print(f"\n  {'Method':<42}  {'ASSE':>6}  {'MAE':>7}")
    print(f"  {'─'*42}  {'─'*6}  {'─'*7}")

    def show(label, yhat_scalar):
        ae = np.abs(Y_test - yhat_scalar)
        print(f"  {label:<42}  {ae.sum():6.3f}  {ae.mean():7.4f}")
        return ae

    ae_mean = show("Marginal mean  [A: GH direct]", yhat_mean)
    ae_medA = show("Marginal median [A: GH CDF]",
                   quantile_A(0.5, TEST_S, theta_tr))
    ae_medB = show("Marginal median [B: GMix closed-form]",
                   quantile_B(0.5, TEST_S, theta_tr))
    print(f"  {'─'*42}  {'─'*6}  {'─'*7}")
    for q in qs:
        show(f"  Delta-quantile q={q:.2f}", dq_vals[q])

    # In-sample posterior (uses y_i — double-dipping reference)
    print(f"  {'─'*42}  {'─'*6}  {'─'*7}")
    fitted_ins = np.array([
        _posterior_mean_mu(Y_test[i], TEST_S, b0, b1, sig, mu_d, sig_d) - sig * EULER_GAMMA
        for i in range(n_test)
    ])
    ae_ins = np.abs(Y_test - fitted_ins)
    print(f"  {'In-sample posterior  (DOUBLE-DIPPING ref)':<42}  {ae_ins.sum():6.3f}  {ae_ins.mean():7.4f}")

    bonus = ae_mean.sum() - ae_ins.sum()
    print(f"\n  Sharpening bonus: {bonus:+.3f}  "
          f"({bonus / ae_mean.sum() * 100:+.1f}% reduction via double-dipping)")

    # ── Per-observation table ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Per-observation breakdown")
    print(f"{'─'*65}")
    print(f"  {'i':>2}  {'y_i':>7}  {'yhat_mean':>10}  {'AE_mean':>8}  {'yhat_ins':>9}  {'AE_ins':>7}")
    print(f"  {'─'*2}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*7}")
    for i in range(n_test):
        print(f"  {i+1:2d}  {Y_test[i]:7.4f}  {yhat_mean:10.4f}  "
              f"{ae_mean[i]:8.4f}  {fitted_ins[i]:9.4f}  {ae_ins[i]:7.4f}")

    # ── Marginal survival curve ────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Marginal survival curve at S={TEST_S}  (re-fitted theta)")
    print(f"{'─'*65}")
    t_vals = np.linspace(Y_test.min() - 0.3, Y_test.max() + 0.3, 12)
    print(f"  {'t':>7}  {'P(Y>t)|A':>10}  {'P(Y>t)|B':>10}  {'cycles':>10}")
    for t in t_vals:
        sA = 1 - marginal_cdf_A(t, TEST_S, theta_tr)
        sB = 1 - marginal_cdf_B(t, TEST_S, theta_tr)
        print(f"  {t:7.4f}  {sA:10.4f}  {sB:10.4f}  {np.exp(t):10.3f}")
    print()
