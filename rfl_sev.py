"""
rfl_sev.py -- SEV + NPMLE semi-parametric RFL
Compare: Normal + NPMLE (existing) vs SEV + NPMLE
Real data n=75, K=2
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328   # E[standard SEV] = -γ_E

# ═══════════════════════════════════════════════════════════
# 共用工具
# ═══════════════════════════════════════════════════════════
def mu_mat(S, deltas, b0, b1):
    diff  = S[:,None] - deltas[None,:]
    valid = diff > 1e-8
    lnd   = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.)
    mu    = np.where(valid, b0 + b1*lnd, -1e30)
    return mu, valid, lnd

# ═══════════════════════════════════════════════════════════
# SEV 機率函數
#   f(y|μ,σ)      = (1/σ) exp(w − e^w),  w = (y−μ)/σ
#   P(Y>y|μ,σ)   = exp(−e^w)
#   E[Y|μ,σ]     = μ − σ γ_E
# ═══════════════════════════════════════════════════════════
def sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def sev_logsf(y, mu, sig):
    """log P(Y>y)"""
    return -np.exp((y - mu) / sig)

def sev_mean(mu, sig):
    return mu - sig * EULER_GAMMA

# ═══════════════════════════════════════════════════════════
# NORMAL + NPMLE  (複製自 rfl_profile.py，用於對照)
# ═══════════════════════════════════════════════════════════
def e_step_normal(Y, S, cens, pis, deltas, b0, b1, sig):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    log_pi = np.log(pis[None,:])
    logw   = np.full((len(Y), len(pis)), -1e30)
    obs    = ~cens
    if obs.any():
        logw[obs] = np.where(valid[obs],
                             log_pi + norm.logpdf(Y[obs,None], mu[obs], sig), -1e30)
    if cens.any():
        a = (Y[cens,None] - mu[cens]) / sig
        logw[cens] = np.where(valid[cens],
                              log_pi + np.log(norm.sf(a).clip(1e-300)), -1e30)
    rm = logw.max(1, keepdims=True)
    e  = np.exp(logw - rm)
    return e / e.sum(1, keepdims=True)

def _m_delta_k(k, Y, S, cens, tau_k, b0, b1, sig, logpdf_fn, logsf_fn):
    S_min = S.min(); obs = ~cens
    def neg_q(dk):
        if dk >= S_min - 1e-6: return 1e10
        diff = S - dk; ok = diff > 1e-8
        if not ok.any(): return 1e10
        mu_k = b0 + b1 * np.log(np.maximum(diff, 1e-300))
        v = 0.
        if obs.any():
            v += (tau_k[obs] * ok[obs] * logpdf_fn(Y[obs], mu_k[obs], sig)).sum()
        if cens.any():
            v += (tau_k[cens] * ok[cens] * logsf_fn(Y[cens], mu_k[cens], sig)).sum()
        return -v
    return minimize_scalar(neg_q, bounds=(0.01, S_min - 0.001), method='bounded').x

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
    A = np.array([[W, S01], [S01, S11]])
    try: b0, b1 = np.linalg.solve(A, [T0, T1])
    except: b0, b1 = b0_old, b1_old
    mu2, valid2, _ = mu_mat(S, deltas, b0, b1)
    w2 = tau * valid2.astype(float)
    ss = (w2[obs] * (Y[obs,None] - mu2[obs])**2).sum()
    wt = w2[obs].sum()
    if cens.any():
        a_c2 = (Y[cens,None] - mu2[cens]) / sig_old
        lam2  = norm.pdf(a_c2) / norm.sf(a_c2).clip(1e-300)
        ss += (w2[cens] * sig_old**2 * (1 + a_c2 * lam2)).sum()
        wt  += w2[cens].sum()
    return b0, b1, max(np.sqrt(ss / max(wt, 1e-8)), 0.01)

def log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig, logpdf_fn, logsf_fn):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    comp = np.where(valid, pis[None,:], 0.)
    ll = 0.; obs = ~cens
    if obs.any():
        dens = np.exp(logpdf_fn(Y[obs,None], mu[obs], sig))
        ll += np.log((comp[obs] * dens).sum(1).clip(1e-300)).sum()
    if cens.any():
        surv = np.exp(logsf_fn(Y[cens,None], mu[cens], sig))
        ll += np.log((comp[cens] * surv).sum(1).clip(1e-300)).sum()
    return ll

def normal_logpdf(y, mu, sig): return norm.logpdf(y, mu, sig)
def normal_logsf(y, mu, sig):  return np.log(norm.sf((y - mu)/sig).clip(1e-300))

def em_full(Y, S, K, delta_init, cens, max_iter, tol, mode):
    """mode: 'normal' or 'sev'"""
    pis = np.ones(K)/K
    deltas = np.sort(np.asarray(delta_init[:K], float))
    obs = ~cens
    lnd0 = np.log(np.maximum(S[obs] - deltas.mean(), 1e-8))
    Xm = np.column_stack([np.ones(obs.sum()), lnd0])
    b = np.linalg.lstsq(Xm, Y[obs], rcond=None)[0]
    b0, b1 = float(b[0]), float(b[1])
    resid = Y[obs] - (b0 + b1 * lnd0)

    if mode == 'normal':
        sig = max(float(np.std(resid)), 0.1)
        lpdf, lsf = normal_logpdf, normal_logsf
        estep = lambda pis_, d_, b0_, b1_, s_: e_step_normal(Y, S, cens, pis_, d_, b0_, b1_, s_)
        mparams = lambda tau_, d_, b0_, b1_, s_: m_step_params_normal(Y, S, cens, tau_, d_, b0_, b1_, s_)
    else:
        sig = max(float(np.std(resid) * np.sqrt(6) / np.pi), 0.1)
        lpdf, lsf = sev_logpdf, sev_logsf
        estep = lambda pis_, d_, b0_, b1_, s_: e_step_sev(Y, S, cens, pis_, d_, b0_, b1_, s_)
        mparams = lambda tau_, d_, b0_, b1_, s_: m_step_params_sev(Y, S, cens, tau_, d_, b0_, b1_, s_)

    ll_prev = -np.inf
    for it in range(max_iter):
        tau = estep(pis, deltas, b0, b1, sig)
        pis = np.maximum(tau.mean(0), 1e-8); pis /= pis.sum()
        for k in range(K):
            deltas[k] = _m_delta_k(k, Y, S, cens, tau[:,k], b0, b1, sig, lpdf, lsf)
        b0, b1, sig = mparams(tau, deltas, b0, b1, sig)
        ll = log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig, lpdf, lsf)
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    return dict(pis=pis, deltas=deltas, b0=b0, b1=b1, sig=sig, ll=ll, iters=it+1, mode=mode)

# ═══════════════════════════════════════════════════════════
# SEV E-step（獨立定義，避免 lambda 引用問題）
# ═══════════════════════════════════════════════════════════
def e_step_sev(Y, S, cens, pis, deltas, b0, b1, sig):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    log_pi = np.log(pis[None,:])
    logw   = np.full((len(Y), len(pis)), -1e30)
    obs    = ~cens
    if obs.any():
        logw[obs] = np.where(valid[obs],
                             log_pi + sev_logpdf(Y[obs,None], mu[obs], sig), -1e30)
    if cens.any():
        logw[cens] = np.where(valid[cens],
                              log_pi + sev_logsf(Y[cens,None], mu[cens], sig), -1e30)
    rm = logw.max(1, keepdims=True)
    e  = np.exp(logw - rm)
    return e / e.sum(1, keepdims=True)

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
    x0 = [b0_old, b1_old, np.log(sig_old)]
    res = minimize(neg_Q, x0, method='BFGS', options={'gtol':1e-6, 'maxiter':300})
    b0, b1, log_sig = res.x
    return b0, b1, max(np.exp(log_sig), 0.01)

# ═══════════════════════════════════════════════════════════
# Profile SE（共用邏輯，傳入 mode）
# ═══════════════════════════════════════════════════════════
def em_G_only(Y, S, cens, b0, b1, sig, pis_init, deltas_init, mode, max_iter=300, tol=1e-9):
    pis = pis_init.copy(); deltas = deltas_init.copy()
    lpdf = normal_logpdf if mode == 'normal' else sev_logpdf
    lsf  = normal_logsf  if mode == 'normal' else sev_logsf
    estep = (lambda: e_step_normal(Y, S, cens, pis, deltas, b0, b1, sig)
             if mode == 'normal'
             else lambda: e_step_sev(Y, S, cens, pis, deltas, b0, b1, sig))
    ll_prev = -np.inf
    for _ in range(max_iter):
        tau = e_step_normal(Y, S, cens, pis, deltas, b0, b1, sig) if mode=='normal' \
              else e_step_sev(Y, S, cens, pis, deltas, b0, b1, sig)
        pis = np.maximum(tau.mean(0), 1e-8); pis /= pis.sum()
        for k in range(len(pis)):
            deltas[k] = _m_delta_k(k, Y, S, cens, tau[:,k], b0, b1, sig, lpdf, lsf)
        ll = log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig, lpdf, lsf)
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    return pis, deltas, ll

def profile_hessian_se(Y, S, cens, res, h=None):
    b0, b1, sig = res['b0'], res['b1'], res['sig']
    pis0, deltas0, mode = res['pis'], res['deltas'], res['mode']
    if h is None: h = np.array([0.05, 0.08, 0.02])
    def pl(db0=0., db1=0., ds=0.):
        s = sig + ds
        if s <= 0: return -1e30
        _, _, ll = em_G_only(Y, S, cens, b0+db0, b1+db1, s, pis0, deltas0, mode)
        return ll
    ll0 = pl()
    e = [np.array([h[0],0,0]), np.array([0,h[1],0]), np.array([0,0,h[2]])]
    H = np.zeros((3,3))
    for i in range(3):
        dp = dict(zip(['db0','db1','ds'], e[i]))
        dm = dict(zip(['db0','db1','ds'], -e[i]))
        H[i,i] = (pl(**dp) - 2*ll0 + pl(**dm)) / h[i]**2
        for j in range(i+1, 3):
            pp = dict(zip(['db0','db1','ds'], e[i]+e[j]))
            pm = dict(zip(['db0','db1','ds'], e[i]-e[j]))
            mp = dict(zip(['db0','db1','ds'],-e[i]+e[j]))
            mm = dict(zip(['db0','db1','ds'],-e[i]-e[j]))
            H[i,j] = H[j,i] = (pl(**pp)-pl(**pm)-pl(**mp)+pl(**mm))/(4*h[i]*h[j])
    try:
        cov = np.linalg.inv(-H)
        se  = np.sqrt(np.abs(np.diag(cov)))
        return se, cov
    except:
        return np.full(3, np.nan), None

# ═══════════════════════════════════════════════════════════
# 實際資料
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# 跑 EM（各 15 個起始點，取最佳 ll）
# ═══════════════════════════════════════════════════════════
results = {}
for mode in ['normal', 'sev']:
    best_ll = -np.inf; best_r = None
    for seed in range(15):
        np.random.seed(seed)
        d = np.sort(np.random.uniform(0.30, 0.62, 2))
        try:
            r = em_full(Y_r, S_r, 2, d, cens_r, max_iter=2000, tol=1e-9, mode=mode)
            if r['ll'] > best_ll: best_ll = r['ll']; best_r = r
        except: pass
    results[mode] = best_r
    print(f"[{mode}] EM done: ll={best_r['ll']:.4f}, iters={best_r['iters']}")

# ═══════════════════════════════════════════════════════════
# Profile SE
# ═══════════════════════════════════════════════════════════
se = {}
for mode in ['normal', 'sev']:
    r = results[mode]
    se_r, _ = profile_hessian_se(Y_r, S_r, cens_r, r)
    se[mode] = se_r
    print(f"[{mode}] Profile SE done")

# ═══════════════════════════════════════════════════════════
# Residuals — two definitions
#   (a) marginal: sum_k pi_k * mu_k(S)            [naive]
#   (b) MAP-conditional: mu_{k*}(S) where k*=argmax tau_ik
#       analogous to INLA's per-obs posterior-mode Delta_hat_i
# ═══════════════════════════════════════════════════════════
fitted_marginal = {}
fitted_map      = {}

for mode in ['normal', 'sev']:
    r = results[mode]
    mu_f, valid_f, _ = mu_mat(S_r, r['deltas'], r['b0'], r['b1'])
    weights = np.where(valid_f, r['pis'][None,:], 0.)

    # (a) marginal: E[Y|S] using location (not mean for SEV)
    fitted_marginal[mode] = (weights * mu_f).sum(1)

    # (b) MAP component per observation → conditional location mu_k*(S_i)
    if mode == 'normal':
        tau = e_step_normal(Y_r, S_r, cens_r, r['pis'], r['deltas'],
                            r['b0'], r['b1'], r['sig'])
    else:
        tau = e_step_sev(Y_r, S_r, cens_r, r['pis'], r['deltas'],
                         r['b0'], r['b1'], r['sig'])
    k_star = tau.argmax(axis=1)                          # MAP component index
    delta_map = r['deltas'][k_star]                      # MAP Delta per obs
    diff_map  = np.maximum(S_r - delta_map, 1e-8)
    fitted_map[mode] = r['b0'] + r['b1'] * np.log(diff_map)  # mu(S, Delta_hat)

# ═══════════════════════════════════════════════════════════
# 輸出結果
# ═══════════════════════════════════════════════════════════
rN, rS = results['normal'], results['sev']
seN, seS = se['normal'], se['sev']

k_par = 6   # (pi1, Delta1, Delta2, b0, b1, sig)
aic = {m: -2*results[m]['ll'] + 2*k_par for m in ['normal','sev']}
bic = {m: -2*results[m]['ll'] + np.log(n)*k_par for m in ['normal','sev']}

mae_marg = {m: np.mean(np.abs(Y_r - fitted_marginal[m])) for m in ['normal','sev']}
mae_map  = {m: np.mean(np.abs(Y_r - fitted_map[m]))      for m in ['normal','sev']}
med_map  = {m: np.median(np.abs(Y_r - fitted_map[m]))    for m in ['normal','sev']}
rmse_map = {m: np.sqrt(np.mean((Y_r - fitted_map[m])**2)) for m in ['normal','sev']}

print()
print("="*62)
print("REAL DATA  K=2  Normal+NPMLE  vs  SEV+NPMLE")
print("="*62)
print(f"\n{'':24} {'Normal+NPMLE':>14} {'SEV+NPMLE':>14}")
print("-"*54)
print(f"  log-lik             {rN['ll']:>14.4f} {rS['ll']:>14.4f}")
print(f"  AIC                 {aic['normal']:>14.4f} {aic['sev']:>14.4f}")
print(f"  BIC                 {bic['normal']:>14.4f} {bic['sev']:>14.4f}")
print(f"  MAE (marginal mu)   {mae_marg['normal']:>14.4f} {mae_marg['sev']:>14.4f}")
print(f"  MAE (MAP Delta_hat) {mae_map['normal']:>14.4f} {mae_map['sev']:>14.4f}  <-- comparable to INLA")
print(f"  MedAE (MAP)         {med_map['normal']:>14.4f} {med_map['sev']:>14.4f}")
print(f"  RMSE  (MAP)         {rmse_map['normal']:>14.4f} {rmse_map['sev']:>14.4f}")
print(f"  EM iters            {rN['iters']:>14d} {rS['iters']:>14d}")
print()

for j, (name, vN, vS) in enumerate(zip(
        ['b0', 'b1', 'sig'],
        [rN['b0'], rN['b1'], rN['sig']],
        [rS['b0'], rS['b1'], rS['sig']])):
    print(f"  {name:<4}                 {vN:>14.5f} {vS:>14.5f}")
    print(f"    SE_profile         {seN[j]:>14.4f} {seS[j]:>14.4f}")

print()
print(f"  pi                  {str(rN['pis'].round(4)):>14} {str(rS['pis'].round(4)):>14}")
print(f"  Delta               {str(rN['deltas'].round(4)):>14} {str(rS['deltas'].round(4)):>14}")

alpha_S = -rS['b1'] / rS['sig']
print(f"\n  alpha=-b1/sig (SEV)           {alpha_S:.4f}")
print(f"  Weibull shape=1/sig (SEV)     {1/rS['sig']:.4f}")

# Per-stress MAP residuals
print(f"\nPer-stress MAP-conditional MAE (comparable to INLA 0.1150):")
print(f"  {'S':>6} {'n':>4} {'Normal':>10} {'SEV':>10} {'INLA':>8} {'Winner':>8}")
stresses = sorted(raw.keys())
# INLA per-stress MAE from README
inla_mae = {0.675:0.0712, 0.750:0.0884, 0.825:0.1165, 0.900:0.1268, 0.950:0.1720}
for s in stresses:
    idx = S_r == s
    mN = np.mean(np.abs(Y_r[idx] - fitted_map['normal'][idx]))
    mS = np.mean(np.abs(Y_r[idx] - fitted_map['sev'][idx]))
    mI = inla_mae[s]
    best = min([('Normal',mN),('SEV',mS),('INLA',mI)], key=lambda x: x[1])[0]
    print(f"  {s:>6.3f} {idx.sum():>4d} {mN:>10.4f} {mS:>10.4f} {mI:>8.4f} {best:>8}")

print("\nDone.")
