"""
rfl_compare_all.py — Holdout: ALL feasible models  (corrected)
===========================================================
Train: S in {0.675, 0.750, 0.825, 0.900}  n=60
Test : S = 0.950                            n=15

Each model is fitted TWICE:
  n=75 fit  -> ASSE_n75 : marginal mean on all 75 obs   (no double-dipping)
  n=60 fit  -> ASSE_te  : marginal mean on 15 holdout    (never seen in fit)

These two columns ARE comparable (same prediction method, different param sets).
readme_ASSE uses posterior mean conditioned on y_i (double-dipping) — listed
for reference only, NOT comparable with the above.

Models:
  [1] Normal+NPMLE  (EM, K by BIC)    readme=33.87
  [2] SEV+NPMLE     (EM, K by BIC)    readme=28.99
  [3] Normal+INLA                      readme= 8.63
  [4] SEV+INLA                         readme= 5.76
  [5] Burr XII+INLA                    readme= 5.74
  [6] Burr+EM-GMM   (Mode A, K=1) *   readme= 4.09
"""
import sys, os
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.stats import norm as scipy_norm
from scipy.special import digamma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EULER_GAMMA = 0.5772156649015328
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)
_SQRT2PI = np.sqrt(2 * np.pi)

# ── Data ──────────────────────────────────────────────────────────────────────
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

TEST_S  = 0.95
mask_tr = S_all != TEST_S
S_train = S_all[mask_tr];  Y_train = Y_all[mask_tr]
S_test  = S_all[~mask_tr]; Y_test  = Y_all[~mask_tr]
n_all   = len(Y_all)
n_train = len(Y_train); n_test = len(Y_test)
cens_tr  = np.zeros(n_train, bool)
cens_all = np.zeros(n_all,   bool)

# ── Common helpers ─────────────────────────────────────────────────────────────
def asse_mae_vec(Y, yhat):
    ae = np.abs(Y - yhat)
    return ae.sum(), ae.mean()

def _mu_mat(S, deltas, b0, b1):
    diff  = S[:,None] - deltas[None,:]
    valid = diff > 1e-8
    ld    = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.)
    mu    = np.where(valid, b0 + b1*ld, -1e30)
    return mu, valid, ld

def sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def npmle_marg_mean(s, pis, deltas, b0, b1, euler_shift=0.):
    diff = s - deltas; ok = diff > 1e-8
    if not ok.any():
        return b0 + b1 * np.log(max(s - deltas.min(), 1e-8)) - euler_shift
    mu_v = b0 + b1 * np.log(np.maximum(diff[ok], 1e-300))
    w = pis[ok] / pis[ok].sum()
    return float(np.dot(w, mu_v)) - euler_shift

# ── P&M E-criterion helpers (rank-matched marginal CDF inversion) ──────────────
# Conditional CDFs
def normal_cdf(y, mu, sig):
    return float(scipy_norm.cdf((y - mu) / sig))

def sev_cdf(y, mu, sig):
    """CDF of SEV (min-extreme-value): F(y) = 1 - exp(-exp(w))."""
    w = (y - mu) / sig
    return float(1.0 - np.exp(-np.exp(min(w, 50.0))))

def burr_cdf(y, mu, sig, a):
    """Burr XII conditional CDF: F(y|delta) = 1 - (a/(a+exp(w)))^a."""
    w = (y - mu) / sig
    log_frac = np.log(a) - np.logaddexp(np.log(a), w)
    return float(1.0 - np.exp(a * log_frac))

# Marginal CDFs (integrate over Delta prior)
def npmle_marg_cdf(y, s, pis, deltas, b0, b1, sig, cond_cdf_fn):
    """Marginal CDF for NPMLE models: F_W = Σ_k π_k F_cond(y|Δ_k)."""
    diff = s - deltas; ok = diff > 1e-8
    if not ok.any():
        mu = b0 + b1 * np.log(max(s - deltas.min(), 1e-8))
        return cond_cdf_fn(y, mu, sig)
    mu_v = b0 + b1 * np.log(np.maximum(diff[ok], 1e-300))
    w = pis[ok] / pis[ok].sum()
    return float(np.dot(w, np.array([cond_cdf_fn(y, mu, sig) for mu in mu_v])))

def inla_gh_marg_cdf(y, s, theta, cond_cdf_fn):
    """Marginal CDF for Normal+INLA or SEV+INLA (GH in log-Delta space)."""
    b0, b1 = theta[0], theta[1]
    sig = np.exp(theta[2])
    mu_d, sig_d = theta[3], np.exp(theta[4])
    z_j = mu_d + sig_d * _GHX
    d_j = np.exp(z_j)
    valid = (d_j > 0) & (d_j < s - 1e-8)
    if not valid.any(): return 0.5
    w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_j[valid])
    return float(np.dot(w_v, np.array([cond_cdf_fn(y, mu, sig) for mu in mu_v])))

def burr_inla_marg_cdf(y, s, theta):
    """Marginal CDF for Burr+INLA (LogNormal prior on Delta, GH in log-Delta)."""
    b0, b1 = theta[0], theta[1]
    sig = np.exp(theta[2]); a = np.exp(theta[3])
    mu_d, sig_d = theta[4], np.exp(theta[5])
    z_j = mu_d + sig_d * _GHX
    d_j = np.exp(z_j)
    valid = (d_j > 0) & (d_j < s - 1e-8)
    if not valid.any(): return 0.5
    w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_j[valid])
    return float(np.dot(w_v, np.array([burr_cdf(y, mu, sig, a) for mu in mu_v])))

def burem_marg_cdf(y, s, b0, b1, sig, a, mu1, sig1):
    """Marginal CDF for Burr+EM-GMM K=1 (Normal prior on Delta, GH in linear-Delta)."""
    d_j = mu1 + sig1 * _GHX
    valid = (d_j > 0) & (d_j < s - 1e-8)
    if not valid.any():
        mu = b0 + b1 * np.log(max(s - mu1, 1e-8))
        return burr_cdf(y, mu, sig, a)
    w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_j[valid])
    return float(np.dot(w_v, np.array([burr_cdf(y, mu, sig, a) for mu in mu_v])))

def marg_quantile(p, s, cdf_fn, y_lo=-20.0, y_hi=20.0):
    """Invert marginal CDF F_W(y;s)=p via brentq. Returns log(ŷ)."""
    try:
        return brentq(lambda y: cdf_fn(y, s) - p, y_lo, y_hi,
                      xtol=1e-5, maxiter=200)
    except ValueError:
        return np.nan

def compute_E(Y_log, S_arr, qtl_fn):
    """
    P&M Table 3 rank-matched prediction error:
      E = Σ_j Σ_i |log(y_{ij}) - log(ŷ_{ij})|
    where ŷ_{ij} = F_W^{-1}((i-0.5)/n_j ; s_j, θ̂)  via qtl_fn(p, s) -> log-scale.
    """
    E_total = 0.0
    for s in np.unique(S_arr):
        idx = S_arr == s
        y_group = np.sort(Y_log[idx])
        n_j = len(y_group)
        for i in range(n_j):
            p = (i + 0.5) / n_j
            yhat = qtl_fn(p, s)
            if not np.isnan(yhat):
                E_total += abs(y_group[i] - yhat)
            else:
                print(f"      [E warn] brentq failed  p={p:.3f}  s={s}")
    return E_total

# ══════════════════════════════════════════════════════════════════════════════
# [1] Normal+NPMLE  (EM, discrete g, K by BIC)
# ══════════════════════════════════════════════════════════════════════════════
def _normal_em_e(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = _mu_mat(S, deltas, b0, b1)
    lt = np.where(valid, np.log(pis[None,:]+1e-300)+scipy_norm.logpdf(Y[:,None],mu,sig), -1e30)
    lt -= lt.max(1, keepdims=True)
    tau = np.exp(lt); tau /= tau.sum(1, keepdims=True)
    return tau

def _normal_em_m_params(Y, S, tau, deltas, b0p, b1p, _):
    mu, valid, ld = _mu_mat(S, deltas, b0p, b1p)
    w = tau*valid; W = w.sum()
    S01=(w*ld).sum(); S11=(w*ld**2).sum()
    T0=(w*Y[:,None]).sum(); T1=(w*ld*Y[:,None]).sum()
    try:
        b0n,b1n = np.linalg.solve([[W,S01],[S01,S11]], [T0,T1])
    except:
        b0n,b1n = b0p, b1p
    mun = np.where(valid, b0n+b1n*ld, 0.)
    sig_n = max(np.sqrt((w*(Y[:,None]-mun)**2).sum()/W), 0.01)
    return b0n, b1n, sig_n

def _em_m_delta(k, Y, S, tau_k, b0, b1, sig, logpdf_fn):
    Sm = S.min()
    def nq(dk):
        if dk >= Sm-1e-6: return 1e10
        diff = S - dk; ok = diff > 1e-8
        if not ok.any(): return 1e10
        mu_k = b0 + b1 * np.log(np.maximum(diff, 1e-300))
        return -(tau_k * ok * logpdf_fn(Y, mu_k, sig)).sum()
    return minimize_scalar(nq, bounds=(0.01, Sm-0.001), method='bounded').x

def _normal_em_ll(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = _mu_mat(S, deltas, b0, b1)
    comp = np.where(valid, pis[None,:]*scipy_norm.pdf(Y[:,None],mu,sig), 0.)
    return np.log(comp.sum(1).clip(1e-300)).sum()

def run_normal_em(Y, S, K, seed=42, max_iter=400, tol=1e-6):
    rng = np.random.default_rng(seed); Sm = S.min()
    best_ll = -np.inf; best = None
    for _ in range(10):
        deltas = np.sort(rng.uniform(0.30, Sm-0.02, K))
        x0 = np.log(np.maximum(S - deltas.mean(), 1e-8))
        Xm = np.column_stack([np.ones(len(Y)), x0])
        bb = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        b0,b1 = bb[0], bb[1]; sig = max(np.std(Y-Xm@bb), 0.1)
        pis = np.ones(K)/K; ll_prev = -np.inf
        for _ in range(max_iter):
            tau = _normal_em_e(Y, S, pis, deltas, b0, b1, sig)
            pis = tau.mean(0); pis = np.maximum(pis, 1e-8); pis /= pis.sum()
            for k in range(K):
                deltas[k] = _em_m_delta(k, Y, S, tau[:,k], b0, b1, sig, scipy_norm.logpdf)
            b0, b1, sig = _normal_em_m_params(Y, S, tau, deltas, b0, b1, sig)
            ll = _normal_em_ll(Y, S, pis, deltas, b0, b1, sig)
            if abs(ll-ll_prev) < tol: break
            ll_prev = ll
        if ll > best_ll:
            best_ll = ll
            best = dict(pis=pis.copy(), deltas=deltas.copy(), b0=b0, b1=b1, sig=sig, ll=ll)
    return best

# ══════════════════════════════════════════════════════════════════════════════
# [2] SEV+NPMLE  (EM, SEV likelihood, K by BIC)
# ══════════════════════════════════════════════════════════════════════════════
def _sev_em_e(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = _mu_mat(S, deltas, b0, b1)
    lp = np.where(valid, np.log(pis[None,:]+1e-300)+sev_logpdf(Y[:,None],mu,sig), -1e30)
    lp -= lp.max(1, keepdims=True)
    tau = np.exp(lp); tau /= tau.sum(1, keepdims=True)
    return tau

def _sev_em_ll(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = _mu_mat(S, deltas, b0, b1)
    lp = np.where(valid, np.log(pis[None,:]+1e-300)+sev_logpdf(Y[:,None],mu,sig), -1e30)
    lmax = lp.max(1, keepdims=True)
    return (np.log(np.exp(lp-lmax).sum(1)) + lmax[:,0]).sum()

def _sev_m_params(Y, S, tau, deltas, b0p, b1p, sig_p):
    def neg_Q(params):
        b0,b1,ls = params; sig = np.exp(ls)
        if sig < 0.01 or sig > 20: return 1e10
        mu, valid, _ = _mu_mat(S, deltas, b0, b1)
        ww = tau * valid
        return -(ww * sev_logpdf(Y[:,None], mu, sig)).sum()
    res = minimize(neg_Q, [b0p,b1p,np.log(sig_p)], method='BFGS',
                   options={'gtol':1e-5, 'maxiter':200})
    b0,b1,ls = res.x
    return b0, b1, max(np.exp(ls), 0.01)

def run_sev_em(Y, S, K, seed=42, max_iter=400, tol=1e-6):
    rng = np.random.default_rng(seed); Sm = S.min()
    best_ll = -np.inf; best = None
    for _ in range(10):
        deltas = np.sort(rng.uniform(0.30, Sm-0.02, K))
        x0 = np.log(np.maximum(S - deltas.mean(), 1e-8))
        Xm = np.column_stack([np.ones(len(Y)), x0])
        bb = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        b0,b1 = bb[0], bb[1]
        sig = max(np.std(Y-Xm@bb)*np.sqrt(6)/np.pi, 0.1)
        pis = np.ones(K)/K; ll_prev = -np.inf
        for _ in range(max_iter):
            tau = _sev_em_e(Y, S, pis, deltas, b0, b1, sig)
            pis = tau.mean(0); pis = np.maximum(pis, 1e-8); pis /= pis.sum()
            for k in range(K):
                deltas[k] = _em_m_delta(k, Y, S, tau[:,k], b0, b1, sig, sev_logpdf)
            b0, b1, sig = _sev_m_params(Y, S, tau, deltas, b0, b1, sig)
            ll = _sev_em_ll(Y, S, pis, deltas, b0, b1, sig)
            if abs(ll-ll_prev) < tol: break
            ll_prev = ll
        if ll > best_ll:
            best_ll = ll
            best = dict(pis=pis.copy(), deltas=deltas.copy(), b0=b0, b1=b1, sig=sig, ll=ll)
    return best

def select_em_K(run_fn, Y, S, n_obs, label):
    best_bic = np.inf; best_K = 1; fits = {}
    for K in [1,2,3,4]:
        r = run_fn(Y, S, K); n_par = 3 + 2*K - 1
        bic = -2*r['ll'] + n_par*np.log(n_obs)
        fits[K] = (r, bic)
        if bic < best_bic: best_bic = bic; best_K = K
    r = fits[best_K][0]
    print(f"    {label}: BIC-best K={best_K}  ll={r['ll']:.4f}  "
          f"b0={r['b0']:.4f}  b1={r['b1']:.4f}  sig={r['sig']:.4f}")
    return r

# ══════════════════════════════════════════════════════════════════════════════
# [3] Normal+INLA  [4] SEV+INLA
# ══════════════════════════════════════════════════════════════════════════════
from rfl_inla import loglik as norm_loglik, _BOUNDS as NORM_BOUNDS
from rfl_sev_inla import loglik as sev_loglik, _BOUNDS as SEV_BOUNDS

SEV_WARM = np.array([-9.3700, -8.5340, np.log(0.1900), -0.6440, np.log(0.0360)])

def fit_fast(loglik_fn, bounds, Y, S, cens, warm=None, seed=42, n_rand=7):
    rng = np.random.default_rng(seed)
    cands = ([] if warm is None else [warm.tolist()]) + \
            [[rng.uniform(lo,hi) for lo,hi in bounds] for _ in range(n_rand)]
    evals = [(loglik_fn(np.array(t), Y, S, cens), t) for t in cands]
    evals.sort(key=lambda x: -x[0])
    best_t = np.array(evals[0][1])
    obj = lambda t: -loglik_fn(t, Y, S, cens)
    rb = minimize(obj, x0=best_t, method='L-BFGS-B', bounds=bounds,
                  options={'ftol':1e-10, 'maxiter':500})
    rn = minimize(obj, x0=rb.x, method='Nelder-Mead',
                  options={'maxiter':5000, 'xatol':1e-6, 'fatol':1e-6})
    return rn.x, float(-rn.fun)

def inla_gh_mean(s, theta, euler_shift=0.):
    """Marginal mean E[Y|S=s] for Normal+INLA or SEV+INLA."""
    b0, b1 = theta[0], theta[1]
    mu_d, sig_d = theta[3], np.exp(theta[4])
    z_j = mu_d + sig_d * _GHX
    d_j = np.exp(z_j)
    valid = (d_j > 0) & (d_j < s - 1e-8)
    w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_j[valid])
    return float(np.dot(w_v, mu_v)) - euler_shift

# ══════════════════════════════════════════════════════════════════════════════
# [5] Burr XII+INLA  (6 params: b0,b1,log_sig,log_a,mu_d,log_sig_d)
# ══════════════════════════════════════════════════════════════════════════════
from rfl_burr_inla import loglik as burr_loglik, _BOUNDS as BURR_BOUNDS

def fit_fast_burr(Y, S, warm=None, seed=42, n_rand=7):
    """Fit Burr+INLA; loglik(theta, Y, S) takes no cens arg."""
    rng = np.random.default_rng(seed)
    cands = ([] if warm is None else [warm.tolist()]) + \
            [[rng.uniform(lo,hi) for lo,hi in BURR_BOUNDS] for _ in range(n_rand)]
    evals = [(burr_loglik(np.array(t), Y, S), t) for t in cands]
    evals.sort(key=lambda x: -x[0])
    best_t = np.array(evals[0][1])
    obj = lambda t: -burr_loglik(t, Y, S)
    rb = minimize(obj, x0=best_t, method='L-BFGS-B', bounds=BURR_BOUNDS,
                  options={'ftol':1e-10, 'maxiter':500})
    rn = minimize(obj, x0=rb.x, method='Nelder-Mead',
                  options={'maxiter':5000, 'xatol':1e-6, 'fatol':1e-6})
    return rn.x, float(-rn.fun)

def burr_gh_mean(s, theta):
    """Marginal mean E[Y|S=s] for Burr+INLA (LogNormal prior on Delta)."""
    b0, b1 = theta[0], theta[1]
    sig = np.exp(theta[2]); a = np.exp(theta[3])
    mu_d, sig_d = theta[4], np.exp(theta[5])
    z_j = mu_d + sig_d * _GHX
    d_j = np.exp(z_j)
    valid = (d_j > 0) & (d_j < s - 1e-8)
    w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_j[valid])
    e_mu = float(np.dot(w_v, mu_v))
    correction = sig * (digamma(a) - np.log(a) + EULER_GAMMA)
    return e_mu - correction

# ══════════════════════════════════════════════════════════════════════════════
# [6] Burr+EM-GMM  (Mode A, K=1)
# ══════════════════════════════════════════════════════════════════════════════
from rfl_burr_em import em_fit as burem_fit

def burem_marg_mean(s, b0, b1, sig, a, mu1, sig1):
    """
    Marginal mean E[Y|S=s] for Burr+EM-GMM K=1.
    Delta ~ N(mu1, sig1^2)  [Normal GMM, not LogNormal]
    -> GH quadrature in linear Delta space: d_j = mu1 + sig1 * x_j
    """
    d_j = mu1 + sig1 * _GHX        # linear delta nodes
    valid = (d_j > 0) & (d_j < s - 1e-8)
    if not valid.any():
        # fallback: point mass at mu1
        return b0 + b1 * np.log(max(s - mu1, 1e-8)) - sig * (digamma(a) - np.log(a) + EULER_GAMMA)
    w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_j[valid])
    e_mu = float(np.dot(w_v, mu_v))
    correction = sig * (digamma(a) - np.log(a) + EULER_GAMMA)
    return e_mu - correction

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    SEP = '=' * 72
    print(SEP)
    print("ALL-MODEL Holdout Comparison  (corrected)")
    print(f"  Train: S in {{0.675,0.750,0.825,0.900}}  n={n_train}")
    print(f"  Test : S=0.95  n={n_test}  (never seen during fit)")
    print(f"  Each model fitted TWICE: n=75 (full) and n=60 (train only)")
    print(SEP)

    results = {}  # name -> (readme, ll_tr, ASSE_n75, MAE_n75, ASSE_te, MAE_te, E_n75)

    def record(name, readme, ll_tr, yh_n75, yh_te_scalar, E_n75=None):
        """
        yh_n75       : array(75,) — marginal mean predictions using n=75 params
        yh_te_scalar : float — marginal mean at S=0.95 using n=60 params
        E_n75        : float — P&M rank-matched prediction error (n=75 fit, all 75 obs)
        """
        yh_te = np.full(n_test, yh_te_scalar)
        a75, m75 = asse_mae_vec(Y_all, yh_n75)
        ate, mte = asse_mae_vec(Y_test, yh_te)
        results[name] = (readme, ll_tr, a75, m75, ate, mte, E_n75)
        e_str = f"  E_n75={E_n75:.3f}" if E_n75 is not None else ""
        print(f"    ll_tr={ll_tr:.4f} | ASSE_n75={a75:.3f} MAE_n75={m75:.4f} "
              f"| ASSE_te={ate:.3f} MAE_te={mte:.4f}{e_str}")

    # ── [1] Normal+NPMLE ──────────────────────────────────────────────────────
    print(f"\n[1] Normal+NPMLE ...")
    print("    Fitting n=75 ...")
    r1_75 = select_em_K(run_normal_em, Y_all,   S_all,   n_all,   "Normal+NPMLE n=75")
    print("    Fitting n=60 ...")
    r1_60 = select_em_K(run_normal_em, Y_train, S_train, n_train, "Normal+NPMLE n=60")
    yh_n75_1 = np.array([npmle_marg_mean(s, r1_75['pis'], r1_75['deltas'],
                                          r1_75['b0'], r1_75['b1']) for s in S_all])
    yh_te_1  = npmle_marg_mean(TEST_S, r1_60['pis'], r1_60['deltas'],
                                r1_60['b0'], r1_60['b1'])
    print("    Computing E_n75 ...")
    _cdf1 = lambda y, s: npmle_marg_cdf(y, s, r1_75['pis'], r1_75['deltas'],
                                         r1_75['b0'], r1_75['b1'], r1_75['sig'], normal_cdf)
    E1 = compute_E(Y_all, S_all, lambda p, s: marg_quantile(p, s, _cdf1))
    record("Normal+NPMLE", 33.87, r1_60['ll'], yh_n75_1, yh_te_1, E1)

    # ── [2] SEV+NPMLE ─────────────────────────────────────────────────────────
    print(f"\n[2] SEV+NPMLE ...")
    print("    Fitting n=75 ...")
    r2_75 = select_em_K(run_sev_em, Y_all,   S_all,   n_all,   "SEV+NPMLE n=75")
    print("    Fitting n=60 ...")
    r2_60 = select_em_K(run_sev_em, Y_train, S_train, n_train, "SEV+NPMLE n=60")
    sev_sh_75 = r2_75['sig'] * EULER_GAMMA
    sev_sh_60 = r2_60['sig'] * EULER_GAMMA
    yh_n75_2 = np.array([npmle_marg_mean(s, r2_75['pis'], r2_75['deltas'],
                                          r2_75['b0'], r2_75['b1'], sev_sh_75) for s in S_all])
    yh_te_2  = npmle_marg_mean(TEST_S, r2_60['pis'], r2_60['deltas'],
                                r2_60['b0'], r2_60['b1'], sev_sh_60)
    print("    Computing E_n75 ...")
    _cdf2 = lambda y, s: npmle_marg_cdf(y, s, r2_75['pis'], r2_75['deltas'],
                                         r2_75['b0'], r2_75['b1'], r2_75['sig'], sev_cdf)
    E2 = compute_E(Y_all, S_all, lambda p, s: marg_quantile(p, s, _cdf2))
    record("SEV+NPMLE", 28.99, r2_60['ll'], yh_n75_2, yh_te_2, E2)

    # ── [3] Normal+INLA ───────────────────────────────────────────────────────
    print(f"\n[3] Normal+INLA ...")
    print("    Fitting n=75 ...")
    th3_75, ll3_75 = fit_fast(norm_loglik, NORM_BOUNDS, Y_all, S_all, cens_all)
    print(f"    n=75: ll={ll3_75:.4f}  b0={th3_75[0]:.4f}  b1={th3_75[1]:.4f}  "
          f"sig={np.exp(th3_75[2]):.4f}")
    print("    Fitting n=60 ...")
    th3_60, ll3_60 = fit_fast(norm_loglik, NORM_BOUNDS, Y_train, S_train, cens_tr)
    print(f"    n=60: ll={ll3_60:.4f}  b0={th3_60[0]:.4f}  b1={th3_60[1]:.4f}  "
          f"sig={np.exp(th3_60[2]):.4f}")
    yh_n75_3 = np.array([inla_gh_mean(s, th3_75, 0.) for s in S_all])
    yh_te_3  = inla_gh_mean(TEST_S, th3_60, 0.)
    print("    Computing E_n75 ...")
    _cdf3 = lambda y, s: inla_gh_marg_cdf(y, s, th3_75, normal_cdf)
    E3 = compute_E(Y_all, S_all, lambda p, s: marg_quantile(p, s, _cdf3))
    record("Normal+INLA", 8.63, ll3_60, yh_n75_3, yh_te_3, E3)

    # ── [4] SEV+INLA ──────────────────────────────────────────────────────────
    print(f"\n[4] SEV+INLA ...")
    print("    Fitting n=75 ...")
    th4_75, ll4_75 = fit_fast(sev_loglik, SEV_BOUNDS, Y_all, S_all, cens_all, warm=SEV_WARM)
    sig4_75 = np.exp(th4_75[2])
    print(f"    n=75: ll={ll4_75:.4f}  b0={th4_75[0]:.4f}  b1={th4_75[1]:.4f}  sig={sig4_75:.4f}")
    print("    Fitting n=60 ...")
    th4_60, ll4_60 = fit_fast(sev_loglik, SEV_BOUNDS, Y_train, S_train, cens_tr, warm=SEV_WARM)
    sig4_60 = np.exp(th4_60[2])
    print(f"    n=60: ll={ll4_60:.4f}  b0={th4_60[0]:.4f}  b1={th4_60[1]:.4f}  sig={sig4_60:.4f}")
    yh_n75_4 = np.array([inla_gh_mean(s, th4_75, sig4_75*EULER_GAMMA) for s in S_all])
    yh_te_4  = inla_gh_mean(TEST_S, th4_60, sig4_60*EULER_GAMMA)
    print("    Computing E_n75 ...")
    _cdf4 = lambda y, s: inla_gh_marg_cdf(y, s, th4_75, sev_cdf)
    E4 = compute_E(Y_all, S_all, lambda p, s: marg_quantile(p, s, _cdf4))
    record("SEV+INLA", 5.76, ll4_60, yh_n75_4, yh_te_4, E4)

    # ── [5] Burr XII+INLA ─────────────────────────────────────────────────────
    print(f"\n[5] Burr XII+INLA ...")
    print("    Fitting n=75 ...")
    burr_warm_75 = np.array([th4_75[0], th4_75[1], th4_75[2],
                              np.log(10.), th4_75[3], th4_75[4]])
    th5_75, ll5_75 = fit_fast_burr(Y_all, S_all, warm=burr_warm_75)
    sig5_75, a5_75 = np.exp(th5_75[2]), np.exp(th5_75[3])
    print(f"    n=75: ll={ll5_75:.4f}  b0={th5_75[0]:.4f}  b1={th5_75[1]:.4f}  "
          f"sig={sig5_75:.4f}  a={a5_75:.3f}")
    print("    Fitting n=60 ...")
    burr_warm_60 = np.array([th4_60[0], th4_60[1], th4_60[2],
                              np.log(10.), th4_60[3], th4_60[4]])
    th5_60, ll5_60 = fit_fast_burr(Y_train, S_train, warm=burr_warm_60)
    sig5_60, a5_60 = np.exp(th5_60[2]), np.exp(th5_60[3])
    print(f"    n=60: ll={ll5_60:.4f}  b0={th5_60[0]:.4f}  b1={th5_60[1]:.4f}  "
          f"sig={sig5_60:.4f}  a={a5_60:.3f}")
    yh_n75_5 = np.array([burr_gh_mean(s, th5_75) for s in S_all])
    yh_te_5  = burr_gh_mean(TEST_S, th5_60)
    print("    Computing E_n75 ...")
    _cdf5 = lambda y, s: burr_inla_marg_cdf(y, s, th5_75)
    E5 = compute_E(Y_all, S_all, lambda p, s: marg_quantile(p, s, _cdf5))
    record("Burr+INLA", 5.74, ll5_60, yh_n75_5, yh_te_5, E5)

    # ── [6] Burr+EM-GMM (Mode A, K=1) ────────────────────────────────────────
    print(f"\n[6] Burr+EM-GMM (Mode A, K=1) ...")
    print("    Fitting n=75 ...")
    r6_75 = burem_fit(Y_all, S_all, K=1, fix_sig=False, fix_a=False, verbose=False)
    print(f"    n=75: ll={r6_75['loglik']:.4f}  b0={r6_75['b0']:.4f}  "
          f"b1={r6_75['b1']:.4f}  sig={r6_75['sig']:.4f}  a={r6_75['a']:.3f}  "
          f"mu1={r6_75['mus'][0]:.4f}  sig1={r6_75['sigs'][0]:.4f}")
    print("    Fitting n=60 ...")
    r6_60 = burem_fit(Y_train, S_train, K=1, fix_sig=False, fix_a=False, verbose=False)
    print(f"    n=60: ll={r6_60['loglik']:.4f}  b0={r6_60['b0']:.4f}  "
          f"b1={r6_60['b1']:.4f}  sig={r6_60['sig']:.4f}  a={r6_60['a']:.3f}  "
          f"mu1={r6_60['mus'][0]:.4f}  sig1={r6_60['sigs'][0]:.4f}")
    yh_n75_6 = np.array([burem_marg_mean(s, r6_75['b0'], r6_75['b1'],
                                          r6_75['sig'], r6_75['a'],
                                          r6_75['mus'][0], r6_75['sigs'][0])
                          for s in S_all])
    yh_te_6  = burem_marg_mean(TEST_S, r6_60['b0'], r6_60['b1'],
                                r6_60['sig'], r6_60['a'],
                                r6_60['mus'][0], r6_60['sigs'][0])
    print("    Computing E_n75 ...")
    _b6, _b16 = r6_75['b0'], r6_75['b1']
    _s6, _a6  = r6_75['sig'], r6_75['a']
    _m6, _sg6 = r6_75['mus'][0], r6_75['sigs'][0]
    _cdf6 = lambda y, s: burem_marg_cdf(y, s, _b6, _b16, _s6, _a6, _m6, _sg6)
    E6 = compute_E(Y_all, S_all, lambda p, s: marg_quantile(p, s, _cdf6))
    record("Burr+EM-GMM*", 4.09, r6_60['loglik'], yh_n75_6, yh_te_6, E6)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("FINAL SUMMARY")
    print(f"  E_n75    : P&M Table 3 rank-matched error  sum|log y_ij - log yhat_ij|")
    print(f"             yhat_ij = F_W^{{-1}}((i-0.5)/n_j ; s_j, theta_hat)  via brentq")
    print(f"             n=75 fit, evaluated on all 75 obs (P&M-comparable metric)")
    print(f"  ASSE_n75 : fit on n=75,  marginal mean, all 75 obs  (no double-dipping)")
    print(f"  ASSE_te  : fit on n=60,  marginal mean, 15 holdout  (S=0.95, never seen)")
    print(f"  readme   : n=75 in-sample ASSE, posterior mean per obs (double-dipping,")
    print(f"             NOT comparable with ASSE_n75 / ASSE_te)")
    print(SEP)
    hdr = (f"  {'Model':<18}  {'readme':>6}  {'ll_tr':>8}  "
           f"{'E_n75':>7}  {'ASSE_n75':>8}  {'MAE_n75':>7}  {'ASSE_te':>7}  {'MAE_te':>6}")
    print(hdr)
    print(f"  {'-'*18}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}")
    for name, (readme, ll_tr, a75, m75, ate, mte, e75) in results.items():
        e_str = f"{e75:7.3f}" if e75 is not None else f"{'N/A':>7}"
        print(f"  {name:<18}  {readme:>6.2f}  {ll_tr:8.4f}  "
              f"{e_str}  {a75:8.3f}  {m75:7.4f}  {ate:7.3f}  {mte:6.4f}")
    print(f"  {'-'*80}")
    print(f"  * Burr+EM-GMM Mode A: sig>=0.15 bound prevents degenerate collapse")
    print(f"  P&M (1999) Normal-Normal concrete data benchmark: E=12.84")
    print(f"  E_n75 and ASSE_n75 are complementary: E uses rank-matched quantiles,")
    print(f"    ASSE uses marginal mean (explains why LOO ASSE≈40 can't rank models)")
    print()
