"""
rfl_sev_ksel.py -- SEV + NPMLE, K=1..4 model selection
Compare BIC and ASSE across K values, and vs Normal+NPMLE and INLA
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

# ── shared helpers ────────────────────────────────────────────────
def mu_mat(S, deltas, b0, b1):
    diff  = S[:,None] - deltas[None,:]
    valid = diff > 1e-8
    lnd   = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.)
    mu    = np.where(valid, b0 + b1*lnd, -1e30)
    return mu, valid, lnd

def sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def sev_logsf(y, mu, sig):
    return -np.exp((y - mu) / sig)

def normal_logpdf(y, mu, sig): return norm.logpdf(y, mu, sig)
def normal_logsf(y, mu, sig):  return np.log(norm.sf((y-mu)/sig).clip(1e-300))

# ── E-step (mode = 'normal' or 'sev') ────────────────────────────
def e_step(Y, S, cens, pis, deltas, b0, b1, sig, mode):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    log_pi = np.log(pis[None,:])
    logw   = np.full((len(Y), len(pis)), -1e30)
    obs    = ~cens
    lpdf   = normal_logpdf if mode == 'normal' else sev_logpdf
    lsf    = normal_logsf  if mode == 'normal' else sev_logsf
    if obs.any():
        logw[obs] = np.where(valid[obs], log_pi + lpdf(Y[obs,None], mu[obs], sig), -1e30)
    if cens.any():
        logw[cens] = np.where(valid[cens], log_pi + lsf(Y[cens,None], mu[cens], sig), -1e30)
    rm = logw.max(1, keepdims=True)
    e  = np.exp(logw - rm)
    return e / e.sum(1, keepdims=True)

# ── M-step Delta_k (shared, numerical) ───────────────────────────
def m_step_delta_k(k, Y, S, cens, tau_k, b0, b1, sig, mode):
    S_min = S.min(); obs = ~cens
    lpdf  = normal_logpdf if mode == 'normal' else sev_logpdf
    lsf   = normal_logsf  if mode == 'normal' else sev_logsf
    def neg_q(dk):
        if dk >= S_min - 1e-6: return 1e10
        diff = S - dk; ok = diff > 1e-8
        if not ok.any(): return 1e10
        mu_k = b0 + b1 * np.log(np.maximum(diff, 1e-300))
        v = 0.
        if obs.any():
            v += (tau_k[obs] * ok[obs] * lpdf(Y[obs], mu_k[obs], sig)).sum()
        if cens.any():
            v += (tau_k[cens] * ok[cens] * lsf(Y[cens], mu_k[cens], sig)).sum()
        return -v
    return minimize_scalar(neg_q, bounds=(0.01, S_min - 0.001), method='bounded').x

# ── M-step params: Normal = WLS, SEV = BFGS ──────────────────────
def m_step_params_normal(Y, S, cens, tau, deltas, b0_old, b1_old, sig_old):
    mu, valid, lnd = mu_mat(S, deltas, b0_old, b1_old)
    w = tau * valid.astype(float)
    Y_aug = np.zeros_like(tau)
    obs = ~cens
    Y_aug[obs] = Y[obs, None]
    if cens.any():
        a_c = (Y[cens,None] - mu[cens]) / sig_old
        lam = norm.pdf(a_c) / norm.sf(a_c).clip(1e-300)
        Y_aug[cens] = mu[cens] + sig_old * lam
    W=w.sum(); S01=(w*lnd).sum(); S11=(w*lnd**2).sum()
    T0=(w*Y_aug).sum(); T1=(w*lnd*Y_aug).sum()
    A = np.array([[W,S01],[S01,S11]])
    try: b0, b1 = np.linalg.solve(A, [T0, T1])
    except: b0, b1 = b0_old, b1_old
    mu2, valid2, _ = mu_mat(S, deltas, b0, b1)
    w2 = tau * valid2.astype(float)
    ss = (w2[obs] * (Y[obs,None] - mu2[obs])**2).sum()
    wt = w2[obs].sum()
    if cens.any():
        a_c2 = (Y[cens,None] - mu2[cens]) / sig_old
        lam2  = norm.pdf(a_c2) / norm.sf(a_c2).clip(1e-300)
        ss += (w2[cens] * sig_old**2 * (1 + a_c2*lam2)).sum()
        wt  += w2[cens].sum()
    return b0, b1, max(np.sqrt(ss / max(wt, 1e-8)), 0.01)

def m_step_params_sev(Y, S, cens, tau, deltas, b0_old, b1_old, sig_old):
    obs = ~cens
    def neg_Q(params):
        b0, b1, log_sig = params
        sig = np.exp(log_sig)
        if sig < 0.01 or sig > 20: return 1e10
        mu, valid, _ = mu_mat(S, deltas, b0, b1)
        ww = tau * valid.astype(float)
        q = 0.
        if obs.any():
            q += (ww[obs] * sev_logpdf(Y[obs,None], mu[obs], sig)).sum()
        if cens.any():
            q += (ww[cens] * sev_logsf(Y[cens,None], mu[cens], sig)).sum()
        return -q
    res = minimize(neg_Q, [b0_old, b1_old, np.log(sig_old)],
                   method='BFGS', options={'gtol':1e-6,'maxiter':300})
    b0, b1, log_sig = res.x
    return b0, b1, max(np.exp(log_sig), 0.01)

# ── log-likelihood ────────────────────────────────────────────────
def log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig, mode):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    comp = np.where(valid, pis[None,:], 0.)
    lpdf = normal_logpdf if mode == 'normal' else sev_logpdf
    lsf  = normal_logsf  if mode == 'normal' else sev_logsf
    ll = 0.; obs = ~cens
    if obs.any():
        ll += np.log((comp[obs] * np.exp(lpdf(Y[obs,None], mu[obs], sig))).sum(1).clip(1e-300)).sum()
    if cens.any():
        ll += np.log((comp[cens] * np.exp(lsf(Y[cens,None], mu[cens], sig))).sum(1).clip(1e-300)).sum()
    return ll

# ── full EM ───────────────────────────────────────────────────────
def em_full(Y, S, K, delta_init, cens, mode, max_iter=500, tol=1e-7):
    pis = np.ones(K)/K
    deltas = np.sort(np.asarray(delta_init[:K], float))
    obs = ~cens
    lnd0 = np.log(np.maximum(S[obs] - deltas.mean(), 1e-8))
    Xm = np.column_stack([np.ones(obs.sum()), lnd0])
    b = np.linalg.lstsq(Xm, Y[obs], rcond=None)[0]
    b0, b1 = float(b[0]), float(b[1])
    resid = Y[obs] - (b0 + b1*lnd0)
    sig = max(float(np.std(resid) * (np.sqrt(6)/np.pi if mode=='sev' else 1.0)), 0.1)

    ll_prev = -np.inf
    for it in range(max_iter):
        tau = e_step(Y, S, cens, pis, deltas, b0, b1, sig, mode)
        pis = np.maximum(tau.mean(0), 1e-8); pis /= pis.sum()
        for k in range(K):
            deltas[k] = m_step_delta_k(k, Y, S, cens, tau[:,k], b0, b1, sig, mode)
        if mode == 'normal':
            b0, b1, sig = m_step_params_normal(Y, S, cens, tau, deltas, b0, b1, sig)
        else:
            b0, b1, sig = m_step_params_sev(Y, S, cens, tau, deltas, b0, b1, sig)
        ll = log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig, mode)
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    return dict(pis=pis, deltas=deltas, b0=b0, b1=b1, sig=sig, ll=ll, iters=it+1)

# ── ASSE: posterior-weighted conditional prediction ───────────────
def compute_asse(Y, S, cens, res, mode):
    tau = e_step(Y, S, cens, res['pis'], res['deltas'], res['b0'], res['b1'], res['sig'], mode)
    mu_f, valid_f, _ = mu_mat(S, res['deltas'], res['b0'], res['b1'])
    # posterior-weighted mu (conditions on y_i)
    f_post = (tau * np.where(valid_f, mu_f, 0.)).sum(1)
    ae = np.abs(Y - f_post)
    return ae.sum(), ae.mean(), np.median(ae)

# ── data ──────────────────────────────────────────────────────────
raw = {
    0.675:[102.95,280.32,339.83,366.9,485.62,658.96,896.33,
           1241.76,1250.2,1329.78,1399.83,1459.14,3249.82,11748.1,11748.1],
    0.75: [6.71,9.93,12.6,15.58,16.19,17.28,18.62,
           20.3,24.9,26.26,27.94,36.35,48.42,50.09,67.34],
    0.825:[1.246,1.258,1.46,1.492,2.4,2.41,2.59,
           2.903,3.33,3.59,3.847,4.11,4.82,5.56,5.598],
    0.9:  [0.201,0.216,0.226,0.252,0.257,0.295,0.311,
           0.342,0.356,0.451,0.457,0.509,0.54,0.68,1.129],
    0.95: [0.037,0.072,0.074,0.076,0.083,0.085,0.105,
           0.109,0.12,0.123,0.143,0.203,0.206,0.217,0.257]
}
S_arr, Y_arr = [], []
for s, ys in raw.items():
    for y in ys: S_arr.append(s); Y_arr.append(np.log(y))
S_r = np.array(S_arr); Y_r = np.array(Y_arr)
cens_r = np.zeros(len(Y_r), bool)
n = len(Y_r)

# ── run K=1..4 for both modes ─────────────────────────────────────
print("="*72)
print("K selection: Normal+NPMLE vs SEV+NPMLE  (8 random starts per K)")
print("="*72)
print(f"{'':18} {'ll':>10} {'#par':>5} {'AIC':>10} {'BIC':>10} {'ASSE':>8} {'MedAE':>7}")
print("-"*72)

INLA_ASSE = 8.63  # reference

best_bic = {'normal': np.inf, 'sev': np.inf}
best_K   = {'normal': None,   'sev': None}

for mode in ['normal', 'sev']:
    for K in [1, 2, 3, 4]:
        best_ll = -np.inf; best_r = None
        for seed in range(8):
            np.random.seed(seed)
            d = np.sort(np.random.uniform(0.30, 0.62, K))
            try:
                r = em_full(Y_r, S_r, K, d, cens_r, mode, max_iter=1000, tol=1e-8)
                if r['ll'] > best_ll:
                    best_ll = r['ll']; best_r = r
            except: pass
        if best_r is None: continue

        # params: b0, b1, sig (3) + K*(pi_k, Delta_k) - 1 constraint = 3 + 2K-1 = 2+2K
        n_par = 2 + 2*K
        aic   = -2*best_r['ll'] + 2*n_par
        bic   = -2*best_r['ll'] + np.log(n)*n_par
        asse, mae, medae = compute_asse(Y_r, S_r, cens_r, best_r, mode)

        if bic < best_bic[mode]:
            best_bic[mode] = bic; best_K[mode] = K

        tag = ' <-- BIC best' if K == best_K.get(mode) else ''  # will fill after loop
        print(f"  {mode:6s}  K={K}     {best_r['ll']:>10.4f} {n_par:>5d} {aic:>10.4f} {bic:>10.4f} {asse:>8.2f} {medae:>7.4f}  pi={best_r['pis'].round(3)}")
    print()

print(f"  INLA (K=cont, 5 par)  reference ASSE={INLA_ASSE:.2f}  MAE=0.1150")
print()
print(f"  Normal BIC-best K: {best_K['normal']}")
print(f"  SEV    BIC-best K: {best_K['sev']}")
print()
print("Done.")
