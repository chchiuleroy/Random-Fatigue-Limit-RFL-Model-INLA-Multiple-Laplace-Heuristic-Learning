"""
rfl_burr.py -- Closed-form Burr XII marginal MLE (no numerical integration)

Model derivation:
  Let U_i = (S_i - Delta_i)^alpha,  alpha = -b1/sigma > 0
  SEV likelihood:  f(y_i|U_i) = (1/sigma)*exp(z_i)*U_i*exp(-c_i*U_i)
                   where z_i=(y_i-b0)/sigma,  c_i=exp(z_i)
  Conjugate prior: U_i ~ Gamma(a, b)  [rate parameterization]

  Posterior:  U_i|y_i ~ Gamma(a+1, b+c_i)         [closed form]
  Marginal:   L_i = (1/sigma)*exp(z_i)*a*b^a / (b+c_i)^(a+1)   [Burr XII]

Score equations at MLE:
  d/db0 = 0  =>  mean(p_i) = 1/(a+1),  p_i = c_i/(b+c_i)
  d/da  = 0  =>  a_hat = -n / sum(log q_i),   q_i = 1-p_i
  d/db  = 0  =>  a = q_bar / p_bar   (consistent with above)
  d/dsig= 0  =>  sigma = -sum(y_i * r_i) / n,  r_i = 1-(a+1)*p_i

Fitted value (posterior mean):
  E[Y_i|y_i] = b0 - sigma*(psi(a+1) - log(b+c_i) + EULER_GAMMA)
  [derived from b1*(1/alpha)=-sigma and SEV mean E[Y|mu,sig]=mu-sig*gamma]

Note: alpha (equiv. b1) is NOT identifiable from this marginal alone.
      The stress-level structure enters only through the y_i distribution.
"""
import numpy as np
from scipy.special import digamma, logsumexp
from scipy.optimize import minimize
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

def _log_bpc(z, log_b):
    """Stable log(b + exp(z)) = logsumexp([log_b, z]) element-wise."""
    a_arr = np.full(len(z), log_b)
    return logsumexp(np.vstack([a_arr, z]), axis=0)

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

# ── Log-likelihood ────────────────────────────────────────────────
def loglik(theta, Y):
    b0, log_sig, log_a, log_b = theta
    sig = np.exp(log_sig)
    a   = np.exp(log_a)
    z   = (Y - b0) / sig
    lbpc = _log_bpc(z, log_b)                # log(b + exp(z)), stable
    ll   = np.sum(-np.log(sig) + z + np.log(a) + a*log_b - (a+1)*lbpc)
    return float(ll) if np.isfinite(ll) else -1e30

# ── ASSE via posterior mean ───────────────────────────────────────
def compute_asse(Y, theta):
    b0, log_sig, log_a, log_b = theta
    sig = np.exp(log_sig)
    a   = np.exp(log_a)
    z   = (Y - b0) / sig
    lbpc = _log_bpc(z, log_b)
    # E[Y_i|y_i] = b0 - sig*(psi(a+1) - log(b+c_i) + EULER_GAMMA)
    fitted = b0 - sig * (digamma(a + 1) - lbpc + EULER_GAMMA)
    ae = np.abs(Y - fitted)
    return ae.sum(), ae.mean(), np.median(ae), ae, fitted

# ── Optimizer: grid -> BFGS -> NM ────────────────────────────────
# sig >= 0.1 prevents degenerate collapse; b0 spans data range
_BOUNDS = [(-5., 15.),
           (np.log(0.1), np.log(10.)),
           (np.log(0.1), np.log(200.)),
           (-6., 6.)]

def optimize(Y):
    best_ll, best_x = -np.inf, None
    for b0 in np.linspace(-3., 12., 10):
        for ls in np.linspace(np.log(0.1), np.log(8.), 7):
            for la in np.linspace(np.log(0.2), np.log(50.), 6):
                for lb in np.linspace(-4., 4., 5):
                    x = [b0, ls, la, lb]
                    ll = loglik(x, Y)
                    if ll > best_ll:
                        best_ll, best_x = ll, x[:]
    print(f"  [Grid ] best ll = {best_ll:.4f}")

    res1 = minimize(lambda t: -loglik(t, Y), x0=best_x,
                    method='L-BFGS-B', bounds=_BOUNDS,
                    options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 5000})
    print(f"  [BFGS ] best ll = {-res1.fun:.4f}")

    res2 = minimize(lambda t: -loglik(t, Y), x0=res1.x,
                    method='Nelder-Mead',
                    options={'maxiter': 60000, 'xatol': 1e-11, 'fatol': 1e-11})
    print(f"  [NM   ] best ll = {-res2.fun:.4f}")
    return res2.x, float(-res2.fun)

# ── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 64)
    print(f"Burr XII Marginal MLE  n={n}  (no numerical integration)")
    print("Prior: U_i = (S_i-Delta_i)^alpha ~ Gamma(a, b)")
    print("=" * 64)

    theta, ll = optimize(Y_r)
    b0, log_sig, log_a, log_b = theta
    sig = np.exp(log_sig)
    a   = np.exp(log_a)
    b   = np.exp(log_b)

    print("\n" + "=" * 64)
    print("FINAL RESULTS")
    print("=" * 64)
    print(f"  b0  = {b0:.5f}    sig = {sig:.5f}")
    print(f"  a   = {a:.5f}    b   = {b:.5f}")
    print(f"  log-lik = {ll:.4f}")

    # Verify score equations (use stable log_b+c)
    z    = (Y_r - b0) / sig
    lbpc = _log_bpc(z, log_b)          # log(b + exp(z))
    lc   = z                           # log(exp(z)) = z
    lq   = log_b - lbpc                # log(q) = log(b/(b+c))
    lp   = lc   - lbpc                 # log(p) = log(c/(b+c))
    p    = np.exp(lp)
    q    = np.exp(lq)
    r    = 1 - (a+1)*p
    print(f"\n  Score checks (target = 0):")
    print(f"    b0 : mean(p) - 1/(a+1) = {p.mean() - 1/(a+1):+.3e}")
    a_check = -n / np.sum(lq)
    print(f"    a  : -n/Σlog(q) - a    = {a_check - a:+.3e}")
    a_b_check = (1 - p.mean()) / p.mean()
    print(f"    b  : q_bar/p_bar - a   = {a_b_check - a:+.3e}")
    sig_check = -np.sum(Y_r * r) / n
    print(f"    sig: -Σ(y*r)/n - sig   = {sig_check - sig:+.3e}")

    asse, mae, medae, ae, fitted = compute_asse(Y_r, theta)
    print(f"\n  ASSE  = {asse:.4f}")
    print(f"  MAE   = {mae:.5f}")
    print(f"  MedAE = {medae:.5f}")

    sev_inla_mae = {0.675:0.0483, 0.75:0.0583, 0.825:0.0755,
                    0.9:0.0765, 0.95:0.1252}
    print(f"\nPer-stress breakdown:")
    print(f"  {'S':>6} {'n':>4} {'ASSE':>8} {'MAE':>8} {'SEV+INLA MAE':>14}")
    for s in stresses:
        idx = S_r == s
        print(f"  {s:6.3f} {idx.sum():4d} {ae[idx].sum():8.4f} "
              f"{ae[idx].mean():8.4f} {sev_inla_mae[s]:14.4f}")

    print()
    print("ASSE Comparison:")
    print(f"  Normal + INLA (9-pt GH)  : 8.63")
    print(f"  SEV MCEM                 : 10.89")
    print(f"  SEV + INLA (9-pt GH)     : 5.76")
    print(f"  Burr XII MLE (this)      : {asse:.2f}")
