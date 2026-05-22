"""
rfl_burr2.py -- Corrected Burr XII marginal MLE (all 5 params restored)

Key fix from rfl_burr.py:
  Prior rate b_S = a / (S_i - mu_Delta)^alpha  (stress-dependent)
  alpha = -b1/sigma

Derivation gives a clean marginal:
  L_i = (a^{a+1}/sigma) * exp(w_i) / (a + exp(w_i))^{a+1}

where  w_i = (y_i - mu_i*) / sigma
       mu_i* = b0 + b1 * log(S_i - mu_Delta)   <- SEV location at E[Delta]

All 5 parameters appear: b0, b1, sigma, a, mu_Delta

Score equations at MLE (p_i = exp(w_i)/(a+exp(w_i)), r_i = 1-(a+1)*p_i):
  b0     : mean(r_i) = 0            -> mean(p_i) = 1/(a+1)
  b1     : sum(r_i * log(delta_i)) = 0   delta_i = S_i - mu_Delta
  sigma  : sum(r_i * w_i) = -n
  mu_Del : sum(r_i / delta_i) = 0
  a      : log(a) + 1/a = mean(log(a + exp(w_i)))

Fitted value (posterior predictive mean):
  yhat_i = mu_i* - sigma * (psi(a+1) - log(a+exp(w_i)) + EULER_GAMMA)
"""
import numpy as np
from scipy.special import digamma, logsumexp
from scipy.optimize import minimize, dual_annealing
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

# ── Data ─────────────────────────────────────────────────────────
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
S_r, Y_r = [], []
for s, ys in _raw.items():
    for y in ys: S_r.append(s); Y_r.append(np.log(y))
S_r = np.array(S_r); Y_r = np.array(Y_r)
stresses = sorted(_raw.keys())
n = len(Y_r)

# ── Core quantities ───────────────────────────────────────────────
def _w(theta, Y, S):
    """w_i = (y_i - mu_i*) / sigma, log_apt = log(a + exp(w_i))"""
    b0, b1, log_sig, log_a, mu_d = theta
    sig   = np.exp(log_sig)
    delta = S - mu_d
    if np.any(delta <= 1e-6):
        return None, None
    mu_star = b0 + b1 * np.log(delta)
    w = (Y - mu_star) / sig
    log_apt = logsumexp(np.vstack([np.full(len(w), log_a), w]), axis=0)
    return w, log_apt

# ── Log-likelihood ────────────────────────────────────────────────
def loglik(theta, Y, S):
    b0, b1, log_sig, log_a, mu_d = theta
    sig = np.exp(log_sig)
    a   = np.exp(log_a)
    w, log_apt = _w(theta, Y, S)
    if w is None:
        return -1e30
    ll = np.sum((a+1)*log_a - np.log(sig) + w - (a+1)*log_apt)
    return float(ll) if np.isfinite(ll) else -1e30

# ── Fitted values ─────────────────────────────────────────────────
def compute_asse(Y, S, theta):
    b0, b1, log_sig, log_a, mu_d = theta
    sig   = np.exp(log_sig)
    a     = np.exp(log_a)
    delta = S - mu_d
    mu_star = b0 + b1 * np.log(delta)
    w, log_apt = _w(theta, Y, S)
    # yhat = mu* - sigma*(psi(a+1) - log(a+exp(w)) + EULER_GAMMA)
    fitted = mu_star - sig * (digamma(a + 1) - log_apt + EULER_GAMMA)
    ae = np.abs(Y - fitted)
    return ae.sum(), ae.mean(), np.median(ae), ae, fitted

# ── Score checks ─────────────────────────────────────────────────
def score_check(theta, Y, S):
    b0, b1, log_sig, log_a, mu_d = theta
    sig   = np.exp(log_sig)
    a     = np.exp(log_a)
    delta = S - mu_d
    w, log_apt = _w(theta, Y, S)
    p = np.exp(w - log_apt)          # exp(w)/(a+exp(w))
    r = 1 - (a+1)*p
    print(f"  b0    : mean(r) = {r.mean():+.3e}  (target 0)")
    print(f"  b1    : sum(r*log(d))/n = {np.mean(r*np.log(delta)):+.3e}  (target 0)")
    print(f"  sigma : sum(r*w)/n + 1 = {np.mean(r*w)+1:+.3e}  (target 0)")
    print(f"  mu_d  : sum(r/d)/n = {np.mean(r/delta):+.3e}  (target 0)")
    a_lhs = np.log(a) + 1/a
    a_rhs = np.mean(log_apt)
    print(f"  a     : log(a)+1/a - mean(log_apt) = {a_lhs-a_rhs:+.3e}  (target 0)")

# ── Optimizer ─────────────────────────────────────────────────────
# mu_d < min(S)=0.675; b1 < 0; a bounded away from inf to avoid degenerate MLE
_BOUNDS = [(-20., 2.),           # b0
           (-25., -0.1),          # b1  (must be negative)
           (np.log(0.05), np.log(5.)),  # log_sig
           (np.log(0.1),  np.log(80.)),  # log_a  (capped to prevent a->inf)
           (0.01, 0.65)]          # mu_d < 0.675 = min(S)

# Warm start from SEV+INLA result
_INLA_X0 = [-9.316, -8.534, np.log(0.190), np.log(2.0), 0.525]

def optimize(Y, S, verbose=True):
    # Multi-start grid
    best_ll, best_x = -np.inf, _INLA_X0[:]
    candidates = [_INLA_X0]
    for b1 in [-8., -5., -3.]:
        for sig in [0.15, 0.25, 0.4]:
            for mu_d in [0.45, 0.52, 0.58]:
                for a in [1., 3., 10.]:
                    b0_try = np.mean(Y) + EULER_GAMMA*sig - b1*np.mean(np.log(S-mu_d))
                    candidates.append([b0_try, b1, np.log(sig), np.log(a), mu_d])

    for x in candidates:
        ll = loglik(x, Y, S)
        if ll > best_ll:
            best_ll, best_x = ll, x[:]
    if verbose:
        print(f"  [Grid ] best ll = {best_ll:.4f}")

    res1 = minimize(lambda t: -loglik(t, Y, S), x0=best_x,
                    method='L-BFGS-B', bounds=_BOUNDS,
                    options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 5000})
    if verbose:
        print(f"  [BFGS ] best ll = {-res1.fun:.4f}")

    res2 = minimize(lambda t: -loglik(t, Y, S), x0=res1.x,
                    method='Nelder-Mead',
                    options={'maxiter': 60000, 'xatol': 1e-11, 'fatol': 1e-11})
    if verbose:
        print(f"  [NM   ] best ll = {-res2.fun:.4f}")

    return res2.x, float(-res2.fun)

# ── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print(f"Burr XII Marginal MLE v2  n={n}  (stress-dependent prior)")
    print("L_i = (a^{a+1}/sigma) * exp(w)/(a+exp(w))^{a+1}")
    print(f"w_i = (y_i - b0 - b1*log(S_i-mu_d)) / sigma")
    print("=" * 65)

    theta, ll = optimize(Y_r, S_r)
    b0, b1, log_sig, log_a, mu_d = theta
    sig = np.exp(log_sig)
    a   = np.exp(log_a)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"  b0    = {b0:.5f}    b1    = {b1:.5f}")
    print(f"  sigma = {sig:.5f}    a     = {a:.5f}")
    print(f"  mu_d  = {mu_d:.5f}   E[Delta] ~ mu_d = {mu_d:.5f}")
    print(f"  log-lik = {ll:.4f}")

    print("\n  Score checks:")
    score_check(theta, Y_r, S_r)

    asse, mae, medae, ae, fitted = compute_asse(Y_r, S_r, theta)
    print(f"\n  ASSE  = {asse:.4f}")
    print(f"  MAE   = {mae:.5f}")
    print(f"  MedAE = {medae:.5f}")

    sev_inla = {0.675:0.0483, 0.75:0.0583, 0.825:0.0755, 0.9:0.0765, 0.95:0.1252}
    print(f"\nPer-stress breakdown:")
    print(f"  {'S':>6} {'n':>4} {'ASSE':>8} {'MAE':>8} {'SEV+INLA MAE':>14}")
    for s in stresses:
        idx = S_r == s
        print(f"  {s:6.3f} {idx.sum():4d} {ae[idx].sum():8.4f} "
              f"{ae[idx].mean():8.4f} {sev_inla[s]:14.4f}")

    print()
    print("ASSE Comparison:")
    print(f"  Normal + INLA (9-pt GH)  : 8.63")
    print(f"  SEV MCEM                 : 10.89")
    print(f"  SEV + INLA (9-pt GH)     : 5.76")
    print(f"  Burr XII MLE v2 (this)   : {asse:.2f}")
    print(f"\n  Note: a={a:.1f} bounded at {np.exp(_BOUNDS[3][1]):.0f}; "
          f"MLE drives a->inf (degenerate) without bound")
