"""
rfl_burr_inla.py -- Burr XII + INLA: 3-level hierarchy

Model:
  Delta_i ~ LogNormal(mu_d, sig_d^2)                       [INLA outer integral]
  U_i | Delta_i ~ Gamma(a, a/(S_i-Delta_i)^alpha)          [Gamma conjugate, closed form]
  Y_i | U_i ~ SEV(b0 - sigma*log(U_i), sigma)              [conditional SEV]

alpha = -b1/sigma > 0

Marginalizing U_i analytically (step 2+3) gives the Burr XII conditional:
  f_Burr(y_i | Delta_i) = (a^{a+1}/sigma) * exp(w_i) / (a + exp(w_i))^{a+1}
  where  w_i(d) = (y_i - mu_i*(d)) / sigma
         mu_i*(d) = b0 + b1 * log(S_i - d)

INLA integrates Delta_i (step 1+Burr):
  L_i = integral f_Burr(y_i | d) * LogNormal(d; mu_d, sig_d) dd

Key: as a -> inf,  f_Burr -> f_SEV  =>  Burr+INLA nests rfl_sev_inla.py exactly.
     as a -> 0,   Burr XII has very heavy tails.

Fitted value (posterior predictive mean of Y_i^new):
  Given Delta_i, posterior U_i | Delta_i, y_i ~ Gamma(a+1, b_S(Delta_i) + c_i)
  E[Y_i^new | Delta_i, y_i] = mu_i*(Delta_i) - sigma*(psi(a+1) - log(a+exp(w_i)) + gamma_E)
  Outer: integrate over posterior p(Delta_i | y_i) via GH quadrature.

6 parameters: b0, b1, log_sig, log_a, mu_d, log_sig_d
"""
import numpy as np
from scipy.special import digamma
from scipy.stats import lognorm
from scipy.optimize import minimize_scalar, minimize
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

# ── Gauss-Hermite setup ───────────────────────────────────────────
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)
_GHE = np.exp(_GHX ** 2 / 2)

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

# ── Burr XII log-density conditional on Delta_i = delta ──────────
def _burr_logpdf(delta, y, s, b0, b1, sig, log_a):
    """
    log f_Burr(y | Delta_i = delta)
    = (a+1)*log(a) - log(sig) + w - (a+1)*log(a + exp(w))
    where w = (y - b0 - b1*log(s-delta)) / sig
    """
    if delta <= 0 or delta >= s - 1e-8:
        return -1e30
    mu_star = b0 + b1 * np.log(s - delta)
    w = (y - mu_star) / sig
    a = np.exp(log_a)
    log_apt = np.logaddexp(log_a, w)   # log(a + exp(w)), numerically stable
    return (a + 1) * log_a - np.log(sig) + w - (a + 1) * log_apt

# ── Log integrand: Burr likelihood x LogNormal prior ─────────────
def _log_h(delta, y, s, b0, b1, sig, log_a, mu_d, sig_d):
    lf = _burr_logpdf(delta, y, s, b0, b1, sig, log_a)
    if lf < -1e25:
        return -1e30
    lg = lognorm.logpdf(delta, s=sig_d, scale=np.exp(mu_d))
    return lf + lg

# ── INLA multiple Laplace per observation ─────────────────────────
def _multi_laplace(y, s, b0, b1, sig, log_a, mu_d, sig_d):
    res = minimize_scalar(
        lambda d: -_log_h(d, y, s, b0, b1, sig, log_a, mu_d, sig_d),
        bounds=(1e-6, s - 1e-4), method='bounded')
    d_hat = res.x
    lh0   = _log_h(d_hat, y, s, b0, b1, sig, log_a, mu_d, sig_d)
    if lh0 < -1e25:
        return -1e30

    eps   = max(d_hat * 0.005, 5e-6)
    lhp   = _log_h(d_hat + eps, y, s, b0, b1, sig, log_a, mu_d, sig_d)
    lhm   = _log_h(d_hat - eps, y, s, b0, b1, sig, log_a, mu_d, sig_d)
    kappa = max(-(lhp - 2 * lh0 + lhm) / eps ** 2, 1e-6)
    sig_t = 1.0 / np.sqrt(kappa)

    d_pts  = d_hat + sig_t * _GHX
    lh_pts = np.array([_log_h(dj, y, s, b0, b1, sig, log_a, mu_d, sig_d) for dj in d_pts])
    log_terms = lh_pts + _GHX ** 2 / 2
    log_shift = log_terms.max()
    L_i = sig_t * np.sum(_GHW * np.exp(log_terms - log_shift)) * np.exp(log_shift)
    return np.log(max(L_i, 1e-300))

# ── Total log-likelihood ──────────────────────────────────────────
def loglik(theta, Y, S):
    b0, b1, log_sig, log_a, mu_d, log_sig_d = theta
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)
    a     = np.exp(log_a)
    if sig < 0.01 or sig_d < 0.005 or a < 0.05:
        return -1e30
    return sum(
        _multi_laplace(Y[i], S[i], b0, b1, sig, log_a, mu_d, sig_d)
        for i in range(len(Y))
    )

# ── Fitted value: E[Y_i^new | y_i] via GH posterior mean ─────────
def _posterior_mean_yhat(y, s, b0, b1, sig, log_a, mu_d, sig_d):
    """
    E[Y_i^new | y_i] = integral E[Y_i^new | Delta_i, y_i] * p(Delta_i | y_i) d(Delta_i)
    where E[Y_i^new | Delta_i, y_i]
        = mu_i*(Delta_i) - sigma*(psi(a+1) - log(a+exp(w_i(Delta_i))) + gamma_E)
    """
    a = np.exp(log_a)
    res = minimize_scalar(
        lambda d: -_log_h(d, y, s, b0, b1, sig, log_a, mu_d, sig_d),
        bounds=(1e-6, s - 1e-4), method='bounded')
    d_hat = res.x
    lh0   = _log_h(d_hat, y, s, b0, b1, sig, log_a, mu_d, sig_d)
    # Fallback if posterior is degenerate
    if lh0 < -1e25:
        mu_fb = b0 + b1 * np.log(max(s - d_hat, 1e-8))
        return mu_fb - sig * EULER_GAMMA

    eps   = max(d_hat * 0.005, 5e-6)
    lhp   = _log_h(d_hat + eps, y, s, b0, b1, sig, log_a, mu_d, sig_d)
    lhm   = _log_h(d_hat - eps, y, s, b0, b1, sig, log_a, mu_d, sig_d)
    kappa = max(-(lhp - 2 * lh0 + lhm) / eps ** 2, 1e-6)
    sig_t = 1.0 / np.sqrt(kappa)

    d_pts  = d_hat + sig_t * _GHX
    lh_pts = np.array([_log_h(dj, y, s, b0, b1, sig, log_a, mu_d, sig_d) for dj in d_pts])
    log_terms = lh_pts + _GHX ** 2 / 2
    log_shift = log_terms.max()
    unnorm = _GHW * np.exp(log_terms - log_shift)
    weights = unnorm / max(unnorm.sum(), 1e-300)

    yhat_pts = []
    for dj in d_pts:
        if dj <= 0 or dj >= s - 1e-8:
            yhat_pts.append(b0 + b1 * np.log(max(s - d_hat, 1e-8)) - sig * EULER_GAMMA)
        else:
            mu_star = b0 + b1 * np.log(s - dj)
            w       = (y - mu_star) / sig
            log_apt = np.logaddexp(log_a, w)
            yhat_pts.append(mu_star - sig * (digamma(a + 1) - log_apt + EULER_GAMMA))
    return float(np.sum(weights * np.array(yhat_pts)))

def compute_asse(Y, S, theta):
    b0, b1, log_sig, log_a, mu_d, log_sig_d = theta
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)
    fitted = np.array([
        _posterior_mean_yhat(Y[i], S[i], b0, b1, sig, log_a, mu_d, sig_d)
        for i in range(len(Y))
    ])
    ae = np.abs(Y - fitted)
    return ae.sum(), ae.mean(), np.median(ae), ae, fitted

# ── Bounds ────────────────────────────────────────────────────────
_BOUNDS = [(-20., -2.),                          # b0
           (-20., -2.),                          # b1
           (np.log(0.05), np.log(2.0)),          # log_sig
           (np.log(0.05), np.log(100.)),         # log_a  (large a -> SEV+INLA limit)
           (-3.0, 1.0),                          # mu_d  (LogNormal location)
           (np.log(0.005), np.log(1.0))]         # log_sig_d

# ── Warm starts ───────────────────────────────────────────────────
# SEV+INLA result: b0=-9.316, b1=-8.534, sig=0.190, mu_d=-0.645, sig_d=0.036
# Try several values of log_a
_SEV_INLA_BASE = [-9.316, -8.534, np.log(0.190), None, -0.645, np.log(0.036)]

def optimize(Y, S, verbose=True):
    best_ll, best_x = -np.inf, None

    # Try multiple starting values for log_a
    for log_a_init in [np.log(0.5), np.log(1.0), np.log(2.0), np.log(5.0), np.log(20.0)]:
        x0 = _SEV_INLA_BASE[:3] + [log_a_init] + _SEV_INLA_BASE[4:]
        ll = loglik(x0, Y, S)
        if ll > best_ll:
            best_ll, best_x = ll, x0[:]

    if verbose:
        a_best = np.exp(best_x[3])
        print(f"  [Warm ] best ll = {best_ll:.4f}  (a_init={a_best:.2f})")

    # BFGS polish
    res1 = minimize(lambda t: -loglik(t, Y, S), x0=best_x,
                    method='L-BFGS-B', bounds=_BOUNDS,
                    options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 500})
    if verbose:
        print(f"  [BFGS ] best ll = {-res1.fun:.4f}")

    # NM polish
    res2 = minimize(lambda t: -loglik(t, Y, S), x0=res1.x,
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    if verbose:
        print(f"  [NM   ] best ll = {-res2.fun:.4f}")

    return res2.x, float(-res2.fun)

# ── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print(f"Burr XII + INLA  n={n}  3-level hierarchy")
    print("  Level 1: Delta_i ~ LogNormal(mu_d, sig_d)  [INLA outer integral]")
    print("  Level 2: U_i | Delta_i ~ Gamma(a, a/delta_i^alpha)  [closed form]")
    print("  Level 3: Y_i | U_i ~ SEV  [conditional likelihood]")
    print("  => f_Burr(y|Delta) closed form; INLA integrates Delta")
    print("  => a->inf recovers rfl_sev_inla.py exactly")
    print("=" * 70)

    theta, ll = optimize(Y_r, S_r)
    b0, b1, log_sig, log_a, mu_d, log_sig_d = theta
    sig   = np.exp(log_sig)
    a     = np.exp(log_a)
    sig_d = np.exp(log_sig_d)
    E_d   = np.exp(mu_d + sig_d**2 / 2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  b0={b0:.5f}  b1={b1:.5f}  sig={sig:.5f}")
    print(f"  a={a:.5f}  mu_d={mu_d:.5f}  sig_d={sig_d:.5f}")
    print(f"  E[Delta]={E_d:.5f}")
    print(f"  log-lik = {ll:.4f}")

    asse, mae, medae, ae, fitted = compute_asse(Y_r, S_r, theta)
    print(f"\n  ASSE  = {asse:.4f}")
    print(f"  MAE   = {mae:.5f}")
    print(f"  MedAE = {medae:.5f}")

    sev_inla = {0.675:0.0483, 0.75:0.0583, 0.825:0.0755, 0.9:0.0765, 0.95:0.1252}
    print(f"\nPer-stress breakdown:")
    print(f"  {'S':>6} {'n':>4} {'ASSE':>8} {'MAE':>8} {'SEV+INLA MAE':>14}")
    for s_val in stresses:
        idx = S_r == s_val
        print(f"  {s_val:6.3f} {idx.sum():4d} {ae[idx].sum():8.4f} "
              f"{ae[idx].mean():8.4f} {sev_inla[s_val]:14.4f}")

    print()
    print("ASSE Comparison:")
    print(f"  Normal + INLA (9-pt GH)  : 8.63")
    print(f"  SEV MCEM                 : 10.89")
    print(f"  Burr XII MLE v2          : 15.92")
    print(f"  SEV + INLA (9-pt GH)     : 5.76  (nested model: a->inf)")
    print(f"  Burr XII + INLA (this)   : {asse:.2f}")
    if a > 50:
        print(f"  Note: a={a:.1f} >> 1, effectively converged to SEV+INLA")
    else:
        print(f"  Note: a={a:.3f}, genuine Gamma overdispersion present")
