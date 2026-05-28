"""
rfl_compare.py — Holdout: SEV+INLA vs Normal+INLA vs EM(Normal)
================================================================
Train: S in {0.675, 0.750, 0.825, 0.900}  n=60
Test : S = 0.950                            n=15  (never seen during fit)

Models:
  [1] SEV+INLA    — f(Y|Delta)=SEV, g(Delta)=LogNormal, GH per-obs
  [2] Normal+INLA — f(Y|Delta)=Normal, g(Delta)=LogNormal, GH per-obs
  [3] EM (Normal) — f(Y|Delta)=Normal, g(Delta)=NPMLE discrete, K by BIC

Prediction (no y_i conditioning):
  INLA models : marginal mean via prior-centred 9-pt GH
  EM          : weighted mean over discrete Delta_k support
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm as scipy_norm, lognorm

# ── GH setup (probabilist's form) ─────────────────────────────────────────────
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)
_SQRT2PI = np.sqrt(2 * np.pi)
EULER_GAMMA = 0.5772156649015328

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
Y_test  = Y_all[~mask_tr]
n_train = len(Y_train);    n_test  = len(Y_test)
cens_tr = np.zeros(n_train, bool)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — SEV+INLA
# ══════════════════════════════════════════════════════════════════════════════
from rfl_sev_inla import (loglik as sev_loglik,
                           _BOUNDS as SEV_BOUNDS,
                           EULER_GAMMA as _EG)

SEV_WARM = np.array([-9.3700, -8.5340, np.log(0.1900), -0.6440, np.log(0.0360)])

def sev_marginal_mean(s, theta):
    b0, b1   = theta[0], theta[1]
    sig      = np.exp(theta[2])
    mu_d, sig_d = theta[3], np.exp(theta[4])
    z_j = mu_d + sig_d * _GHX
    d_j = np.exp(z_j); valid = (d_j > 0) & (d_j < s - 1e-8)
    d_v = d_j[valid]; w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_v)
    return float(np.dot(w_v, mu_v)) - sig * EULER_GAMMA

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Normal+INLA
# ══════════════════════════════════════════════════════════════════════════════
from rfl_inla import (loglik as norm_loglik,
                      _BOUNDS as NORM_BOUNDS)

def norm_marginal_mean(s, theta):
    b0, b1      = theta[0], theta[1]
    mu_d, sig_d = theta[3], np.exp(theta[4])
    z_j = mu_d + sig_d * _GHX
    d_j = np.exp(z_j); valid = (d_j > 0) & (d_j < s - 1e-8)
    d_v = d_j[valid]; w_v = _GHW[valid] / _SQRT2PI
    mu_v = b0 + b1 * np.log(s - d_v)
    return float(np.dot(w_v, mu_v))   # Normal: E[Y]=mu, no Euler shift

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — EM (Normal + NPMLE discrete g(Delta))
# ══════════════════════════════════════════════════════════════════════════════
def _mu_mat(S, deltas, b0, b1):
    diff  = S[:,None] - deltas[None,:]
    valid = diff > 1e-8
    ld    = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.0)
    mu    = np.where(valid, b0 + b1 * ld, -1e30)
    return mu, valid, ld

def _em_e(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = _mu_mat(S, deltas, b0, b1)
    lt = np.where(valid, np.log(pis[None,:]+1e-300) +
                  scipy_norm.logpdf(Y[:,None], mu, sig), -1e30)
    lt -= lt.max(axis=1, keepdims=True)
    tau = np.exp(lt); tau /= tau.sum(axis=1, keepdims=True)
    return tau

def _em_m_params(Y, S, tau, deltas, b0p, b1p, _sigp):
    mu, valid, ld = _mu_mat(S, deltas, b0p, b1p)
    w = tau * valid
    W = w.sum()
    S00 = W; S01 = (w*ld).sum(); S11 = (w*ld**2).sum()
    T0  = (w*Y[:,None]).sum(); T1  = (w*ld*Y[:,None]).sum()
    try:
        b0n, b1n = np.linalg.solve([[S00,S01],[S01,S11]], [T0,T1])
    except Exception:
        b0n, b1n = b0p, b1p
    mu_n = np.where(valid, b0n + b1n*ld, 0.0)
    sig_n = max(np.sqrt((w*(Y[:,None]-mu_n)**2).sum()/W), 0.01)
    return b0n, b1n, sig_n

def _em_m_delta(k, Y, S, tau_k, b0, b1, sig):
    Sm = S.min()
    def nq(dk):
        if dk >= Sm-1e-6: return 1e10
        diff = S - dk; valid = diff > 1e-8
        if not valid.any(): return 1e10
        mu = b0 + b1*np.log(np.maximum(diff,1e-300))
        return -(tau_k * valid * scipy_norm.logpdf(Y, mu, sig)).sum()
    return minimize_scalar(nq, bounds=(0.01, Sm-0.001), method='bounded').x

def _em_ll(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = _mu_mat(S, deltas, b0, b1)
    comp = np.where(valid, pis[None,:]*scipy_norm.pdf(Y[:,None], mu, sig), 0.0)
    return np.log(comp.sum(axis=1).clip(1e-300)).sum()

def run_em(Y, S, K, seed=42, max_iter=400, tol=1e-6):
    rng = np.random.default_rng(seed)
    Sm  = S.min()
    best_ll = -np.inf; best = None
    for trial in range(10):
        deltas = np.sort(rng.uniform(0.30, Sm-0.02, K))
        x0 = np.log(np.maximum(S - deltas.mean(), 1e-8))
        Xm = np.column_stack([np.ones(len(Y)), x0])
        bb = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        b0, b1 = bb[0], bb[1]
        sig = max(np.std(Y - Xm@bb), 0.1)
        pis = np.ones(K)/K
        ll_prev = -np.inf
        for _ in range(max_iter):
            tau = _em_e(Y, S, pis, deltas, b0, b1, sig)
            pis = tau.mean(0); pis = np.maximum(pis,1e-8); pis /= pis.sum()
            for k in range(K):
                deltas[k] = _em_m_delta(k, Y, S, tau[:,k], b0, b1, sig)
            b0, b1, sig = _em_m_params(Y, S, tau, deltas, b0, b1, sig)
            ll = _em_ll(Y, S, pis, deltas, b0, b1, sig)
            if abs(ll-ll_prev) < tol: break
            ll_prev = ll
        if ll > best_ll:
            best_ll = ll
            best = dict(pis=pis.copy(), deltas=deltas.copy(),
                        b0=b0, b1=b1, sig=sig, ll=ll)
    return best

def em_marg_mean(s, pis, deltas, b0, b1):
    diff = s - deltas; valid = diff > 1e-8
    if not valid.any(): return np.nan
    mu_v = b0 + b1 * np.log(np.maximum(diff[valid], 1e-300))
    return float(np.dot(pis[valid]/pis[valid].sum(), mu_v))

# ══════════════════════════════════════════════════════════════════════════════
# Fast fitter: L-BFGS-B → Nelder-Mead
# ══════════════════════════════════════════════════════════════════════════════
def fit_fast(loglik_fn, bounds, Y, S, cens, warm=None, seed=42, n_rand=7):
    rng = np.random.default_rng(seed)
    cands = ([] if warm is None else [warm.tolist()]) + \
            [[rng.uniform(lo,hi) for lo,hi in bounds] for _ in range(n_rand)]
    evals = [(loglik_fn(np.array(t), Y, S, cens), t) for t in cands]
    evals.sort(key=lambda x: -x[0])
    best_t = np.array(evals[0][1])
    obj = lambda t: -loglik_fn(t, Y, S, cens)
    res_b = minimize(obj, x0=best_t, method='L-BFGS-B', bounds=bounds,
                     options={'ftol':1e-10,'maxiter':500})
    res_n = minimize(obj, x0=res_b.x, method='Nelder-Mead',
                     options={'maxiter':5000,'xatol':1e-6,'fatol':1e-6})
    return res_n.x, float(-res_n.fun)

# ══════════════════════════════════════════════════════════════════════════════
# ASSE / MAE helpers
# ══════════════════════════════════════════════════════════════════════════════
def asse_mae(Y_true, yhat_scalar):
    ae = np.abs(Y_true - yhat_scalar)
    return ae.sum(), ae.mean()

def asse_mae_vec(Y_true, yhat_vec):
    ae = np.abs(Y_true - yhat_vec)
    return ae.sum(), ae.mean()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    SEP = '=' * 70
    print(SEP)
    print("Holdout comparison: SEV+INLA  vs  Normal+INLA  vs  EM(Normal)")
    print(f"  Train: S in {{0.675,0.750,0.825,0.900}}  n={n_train}")
    print(f"  Test : S={TEST_S}  n={n_test}  (completely held out)")
    print(SEP)

    results = {}  # name -> (ll_train, asse_tr, mae_tr, asse_te, mae_te)

    # ── [1] SEV+INLA ──────────────────────────────────────────────────────────
    print(f"\n[1] SEV+INLA ...")
    th_sev, ll_sev = fit_fast(sev_loglik, SEV_BOUNDS, Y_train, S_train, cens_tr,
                               warm=SEV_WARM)
    b0s,b1s,sigs = th_sev[0],th_sev[1],np.exp(th_sev[2])
    mu_ds,sig_ds = th_sev[3],np.exp(th_sev[4])
    print(f"    ll={ll_sev:.4f}  b0={b0s:.4f}  b1={b1s:.4f}  "
          f"sig={sigs:.4f}  mu_d={mu_ds:.4f}  sig_d={sig_ds:.4f}")

    yh_sev_tr = np.array([sev_marginal_mean(s, th_sev) for s in S_train])
    at_s, mt_s = asse_mae_vec(Y_train, yh_sev_tr)
    yh_sev_te  = sev_marginal_mean(TEST_S, th_sev)
    ae_s, me_s = asse_mae(Y_test, yh_sev_te)
    results['SEV+INLA'] = (ll_sev, at_s, mt_s, ae_s, me_s)
    print(f"    Train: ASSE={at_s:.3f}  MAE={mt_s:.4f}  |  "
          f"Test:  ASSE={ae_s:.3f}  MAE={me_s:.4f}")

    # ── [2] Normal+INLA ───────────────────────────────────────────────────────
    print(f"\n[2] Normal+INLA ...")
    th_norm, ll_norm = fit_fast(norm_loglik, NORM_BOUNDS, Y_train, S_train, cens_tr)
    b0n,b1n,sign = th_norm[0],th_norm[1],np.exp(th_norm[2])
    mu_dn,sig_dn = th_norm[3],np.exp(th_norm[4])
    print(f"    ll={ll_norm:.4f}  b0={b0n:.4f}  b1={b1n:.4f}  "
          f"sig={sign:.4f}  mu_d={mu_dn:.4f}  sig_d={sig_dn:.4f}")

    yh_norm_tr = np.array([norm_marginal_mean(s, th_norm) for s in S_train])
    at_n, mt_n = asse_mae_vec(Y_train, yh_norm_tr)
    yh_norm_te  = norm_marginal_mean(TEST_S, th_norm)
    ae_n, me_n = asse_mae(Y_test, yh_norm_te)
    results['Normal+INLA'] = (ll_norm, at_n, mt_n, ae_n, me_n)
    print(f"    Train: ASSE={at_n:.3f}  MAE={mt_n:.4f}  |  "
          f"Test:  ASSE={ae_n:.3f}  MAE={me_n:.4f}")

    # ── [3] EM (Normal+NPMLE, K by BIC) ──────────────────────────────────────
    print(f"\n[3] EM (Normal + NPMLE Δ) ...")
    best_bic = np.inf; best_K = 1; em_fits = {}
    for K in [1, 2, 3, 4]:
        r = run_em(Y_train, S_train, K)
        n_par = 3 + 2*K - 1          # b0,b1,sig + (K-1) free pi + K deltas
        bic   = -2*r['ll'] + n_par * np.log(n_train)
        aic   = -2*r['ll'] + 2*n_par
        em_fits[K] = (r, bic)
        flag = " <-- BIC best" if bic < best_bic else ""
        print(f"    K={K}  ll={r['ll']:.4f}  #par={n_par}  "
              f"BIC={bic:.3f}  AIC={aic:.3f}{flag}")
        if bic < best_bic:
            best_bic = bic; best_K = K
    r_em = em_fits[best_K][0]
    print(f"    Selected K={best_K}  b0={r_em['b0']:.4f}  b1={r_em['b1']:.4f}  "
          f"sig={r_em['sig']:.4f}")
    print(f"    deltas={np.round(r_em['deltas'],5)}  pis={np.round(r_em['pis'],4)}")

    yh_em_tr = np.array([em_marg_mean(s, r_em['pis'], r_em['deltas'],
                                       r_em['b0'], r_em['b1']) for s in S_train])
    at_e, mt_e = asse_mae_vec(Y_train, yh_em_tr)
    yh_em_te   = em_marg_mean(TEST_S, r_em['pis'], r_em['deltas'],
                               r_em['b0'], r_em['b1'])
    ae_e, me_e = asse_mae(Y_test, yh_em_te)
    results[f'EM(K={best_K})'] = (r_em['ll'], at_e, mt_e, ae_e, me_e)
    print(f"    Train: ASSE={at_e:.3f}  MAE={mt_e:.4f}  |  "
          f"Test:  ASSE={ae_e:.3f}  MAE={me_e:.4f}")
    print(f"    EM predicted mean at S={TEST_S}: {yh_em_te:.4f}  "
          f"(actual mean: {Y_test.mean():.4f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SUMMARY TABLE")
    print(f"{'─'*70}")
    hdr = f"  {'Model':<18}  {'ll_train':>10}  {'ASSE_tr':>7}  {'MAE_tr':>6}  {'ASSE_te':>7}  {'MAE_te':>6}"
    print(hdr)
    print(f"  {'─'*18}  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*6}")
    for name, (ll, at, mt, ae, me) in results.items():
        print(f"  {name:<18}  {ll:10.4f}  {at:7.3f}  {mt:6.4f}  {ae:7.3f}  {me:6.4f}")
    print(f"{'─'*70}")
    print(f"  (Test = S={TEST_S} group, completely held out during fitting)")
    print(f"  ASSE = sum of |y_i - yhat|;  MAE = ASSE / n")
    print()
