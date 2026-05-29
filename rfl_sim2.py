"""
Semi-parametric RFL — Corrected Simulation Study
Fixes:
  1. True delta=(0.40,0.55) for well-separated components
  2. Censored M-step uses augmented data (E[Y|Y>c]) for b0,b1
  3. SE via numerical Hessian of observed log-lik (robust)
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
import warnings; warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# Core helpers
# ════════════════════════════════════════════════════════════════
def mu_mat(S, deltas, b0, b1):
    diff  = S[:,None] - deltas[None,:]
    valid = diff > 1e-8
    lnd   = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.)
    mu    = np.where(valid, b0 + b1*lnd, -1e30)
    return mu, valid, lnd

def e_step_full(Y, S, cens, pis, deltas, b0, b1, sig):
    """
    E-step handling uncensored + right-censored (Type I).
    cens: bool array (True = right-censored at Y_i)
    """
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    n, K = len(Y), len(pis)
    log_pi = np.log(pis[None,:])
    logw = np.full((n, K), -1e30)

    # uncensored: log f(y_i | comp k)
    obs = ~cens
    if obs.any():
        logw[obs] = np.where(valid[obs],
                             log_pi + norm.logpdf(Y[obs,None], mu[obs], sig),
                             -1e30)
    # right-censored: log P(Y_i > c_i | comp k) = log Phi^c((c_i-mu)/sig)
    if cens.any():
        a = (Y[cens,None] - mu[cens]) / sig
        logw[cens] = np.where(valid[cens],
                              log_pi + np.log(norm.sf(a).clip(1e-300)),
                              -1e30)
    rm   = logw.max(1, keepdims=True)
    e    = np.exp(logw - rm)
    tau  = e / e.sum(1, keepdims=True)
    return tau

def m_step_delta(k, Y, S, cens, tau_k, b0, b1, sig):
    S_min = S.min()
    obs   = ~cens
    def neg_q(dk):
        if dk >= S_min - 1e-6: return 1e10
        diff = S - dk; ok = diff > 1e-8
        if not ok.any(): return 1e10
        mu_k = b0 + b1*np.log(np.maximum(diff, 1e-300))
        val  = (tau_k * ok * norm.logpdf(Y, mu_k, sig))[obs].sum()
        if cens.any():
            a = (Y[cens] - mu_k[cens]) / sig
            val += (tau_k[cens] * ok[cens] * np.log(norm.sf(a).clip(1e-300))).sum()
        return -val
    res = minimize_scalar(neg_q, bounds=(0.01, S_min-0.001), method='bounded')
    return res.x

def m_step_params(Y, S, cens, tau, deltas, b0_old, b1_old, sig_old):
    """
    Update b0, b1, sigma using augmented data.
    Right-censored: replace Y_i with E[Y_i | Y_i > c_i, Delta_k]
    """
    mu, valid, lnd = mu_mat(S, deltas, b0_old, b1_old)
    w   = tau * valid.astype(float)   # (n,K)
    obs = ~cens

    # augmented Y: uncensored use Y_i, censored impute E[Y|Y>c, k]
    Y_aug = np.zeros_like(tau)        # (n,K)
    Y_aug[obs]  = Y[obs, None]
    if cens.any():
        a_c = (Y[cens, None] - mu[cens]) / sig_old
        lam = norm.pdf(a_c) / norm.sf(a_c).clip(1e-300)   # inverse Mills
        Y_aug[cens] = mu[cens] + sig_old * lam

    W   = w.sum()
    S01 = (w * lnd).sum()
    S11 = (w * lnd**2).sum()
    T0  = (w * Y_aug).sum()
    T1  = (w * lnd * Y_aug).sum()
    A   = np.array([[W, S01],[S01, S11]])
    try:
        b0, b1 = np.linalg.solve(A, [T0, T1])
    except np.linalg.LinAlgError:
        b0, b1 = b0_old, b1_old

    # sigma: augment second moment for censored
    mu2, valid2, _ = mu_mat(S, deltas, b0, b1)
    w2 = tau * valid2.astype(float)
    ss = (w2[obs] * (Y[obs,None] - mu2[obs])**2).sum()
    wt = w2[obs].sum()
    if cens.any():
        a_c2 = (Y[cens,None] - mu2[cens]) / sig_old
        lam2 = norm.pdf(a_c2) / norm.sf(a_c2).clip(1e-300)
        # E[(Y-mu)^2|Y>c] = sig^2*(1 + a*lam)  ... see e.g. Greene (2003)
        ss += (w2[cens] * sig_old**2 * (1 + a_c2*lam2)).sum()
        wt += w2[cens].sum()
    sig = max(np.sqrt(ss / max(wt, 1e-8)), 0.01)
    return b0, b1, sig

def log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    obs = ~cens
    comp = np.where(valid, pis[None,:], 0.)
    ll   = 0.
    if obs.any():
        ll += np.log((comp[obs] * norm.pdf(Y[obs,None], mu[obs], sig)).sum(1).clip(1e-300)).sum()
    if cens.any():
        a = (Y[cens,None] - mu[cens]) / sig
        ll += np.log((comp[cens] * norm.sf(a)).sum(1).clip(1e-300)).sum()
    return ll

def em_rfl(Y, S, K, delta_init, cens=None, max_iter=500, tol=1e-7):
    if cens is None:
        cens = np.zeros(len(Y), dtype=bool)
    pis    = np.ones(K)/K
    deltas = np.sort(np.asarray(delta_init[:K], float))
    obs    = ~cens
    lnd0   = np.log(np.maximum(S[obs] - deltas.mean(), 1e-8))
    Xm     = np.column_stack([np.ones(obs.sum()), lnd0])
    b      = np.linalg.lstsq(Xm, Y[obs], rcond=None)[0]
    b0, b1 = float(b[0]), float(b[1])
    sig    = max(float(np.std(Y[obs] - Xm@b)), 0.1)

    ll_prev = -np.inf
    for it in range(max_iter):
        tau    = e_step_full(Y, S, cens, pis, deltas, b0, b1, sig)
        pis    = np.maximum(tau.mean(0), 1e-8); pis /= pis.sum()
        for k in range(K):
            deltas[k] = m_step_delta(k, Y, S, cens, tau[:,k], b0, b1, sig)
        b0, b1, sig = m_step_params(Y, S, cens, tau, deltas, b0, b1, sig)
        ll = log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig)
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    return dict(pis=pis, deltas=deltas, b0=b0, b1=b1, sig=sig, ll=ll, iters=it+1)

def louis_se(Y, S, cens, res):
    """
    Louis (1982): I_obs = I_c - I_mis
    I_mis = sum_i sum_k tau_ik g_ik g_ik^T  -  sum_i s_i s_i^T
    """
    pis, deltas, b0, b1, sig = (res['pis'], res['deltas'],
                                 res['b0'],  res['b1'],   res['sig'])
    mu, valid, lnd = mu_mat(S, deltas, b0, b1)
    tau = e_step_full(Y, S, cens, pis, deltas, b0, b1, sig)  # (n,K)

    r   = (Y[:,None] - mu) / sig    # (n,K)
    w   = tau * valid.astype(float)

    # per-component gradients g_ik = [r/sig, r*x/sig, (r^2-1)/sig]
    g0 = r / sig                    # (n,K)
    g1 = r * lnd / sig
    g2 = (r**2 - 1) / sig

    # Complete-data information I_c (negated expected Hessian)
    I_c = np.array([
        [(w / sig**2).sum(),           (w*lnd/sig**2).sum(),     0.],
        [(w*lnd/sig**2).sum(),         (w*lnd**2/sig**2).sum(),  0.],
        [0.,                           0.,              (w*(3*r**2-1)/sig**2).sum()]
    ])

    # E[g_ik g_ik^T | y_i] = sum_k tau_ik * g_ik g_ik^T
    G = np.stack([g0, g1, g2], axis=2)  # (n,K,3)
    E_ggT = np.einsum('nkp,nkq,nk->pq', G, G, tau)

    # observed score s_i = sum_k tau_ik g_ik
    s = np.einsum('nkp,nk->np', G, tau)  # (n,3)
    sum_ssT = s.T @ s

    I_mis  = E_ggT - sum_ssT
    I_obs  = I_c - I_mis
    try:
        cov = np.linalg.inv(I_obs)
        return np.sqrt(np.abs(np.diag(cov)))
    except:
        return np.full(3, np.nan)

# ════════════════════════════════════════════════════════════════
# STEP 1 — Real data K=2 (corrected EM, numerical SE)
# ════════════════════════════════════════════════════════════════
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
S_real = np.array(S_arr); Y_real = np.array(Y_arr); n_r = len(Y_real)
cens_real = np.zeros(n_r, dtype=bool)

print("="*60)
print("STEP 1  Real data  K=2  (corrected, max_iter=2000)")
print("="*60)
best_ll=-np.inf; best_res=None
for seed in range(15):
    np.random.seed(seed)
    d_init = np.sort(np.random.uniform(0.30,0.62,2))
    try:
        r = em_rfl(Y_real,S_real,2,d_init,max_iter=2000,tol=1e-9)
        if r['ll']>best_ll: best_ll=r['ll']; best_res=r
    except: pass

r  = best_res
p2 = 3+2*2-1
bic= -2*r['ll']+p2*np.log(n_r)
print(f"iters={r['iters']}  ll={r['ll']:.5f}  BIC={bic:.4f}")
print(f"b0={r['b0']:.5f}  b1={r['b1']:.5f}  sigma={r['sig']:.5f}")
print(f"pi ={r['pis'].round(5)}  D={r['deltas'].round(6)}")
se = louis_se(Y_real, S_real, cens_real, r)
print(f"SE (num Hess): b0={se[0]:.4f}  b1={se[1]:.4f}  sigma={se[2]:.4f}")
for name, val, s in zip(['b0','b1','sigma'],[r['b0'],r['b1'],r['sig']],se):
    print(f"  95%CI {name}: [{val-1.96*s:.4f}, {val+1.96*s:.4f}]")

# ════════════════════════════════════════════════════════════════
# STEP 2 — Simulation study
# TRUE: K=2, delta=(0.40,0.55) [well-separated], pi=(0.7,0.3)
# ════════════════════════════════════════════════════════════════
TRUE = dict(b0=-9.1, b1=-8.0, sig=0.60,
            pis=np.array([0.7,0.3]), deltas=np.array([0.40,0.55]))
STRESS = np.array([0.675,0.75,0.825,0.9,0.95])
R      = 200

def simulate(m, cr, rng):
    b0,b1,sig = TRUE['b0'],TRUE['b1'],TRUE['sig']
    pis,deltas= TRUE['pis'],TRUE['deltas']
    Y_,S_,C_  = [],[],[]
    for s in STRESS:
        dk  = rng.choice(deltas, size=m, p=pis)
        mu  = b0 + b1*np.log(np.maximum(s-dk,1e-8))
        y   = rng.normal(mu,sig)
        ct  = np.quantile(y,1-cr) if cr>0 else np.inf
        csd = y>ct
        Y_.extend(np.where(csd,ct,y)); S_.extend([s]*m); C_.extend(csd)
    return np.array(Y_),np.array(S_),np.array(C_,bool)

TRUE_VALS = np.array([TRUE['b0'],TRUE['b1'],TRUE['sig']])
NAMES     = ['b0','b1','sigma']

print()
print("="*60)
print("STEP 2  Simulation  (True K=2, delta=(0.40,0.55))")
print(f"  b0={TRUE['b0']}, b1={TRUE['b1']}, sigma={TRUE['sig']}")
print(f"  pi={TRUE['pis']}, R={R}")
print("="*60)
print(f"{'Scenario':<24} | {'Param':<6} | {'Bias':>8} | {'RMSE':>8} | {'Cvg95':>7} | {'SE_avg':>7}")
print("-"*72)

scenarios = [
    (10, 0.0, 'n=50 , cens=0%'),
    (15, 0.0, 'n=75 , cens=0%'),
    (20, 0.0, 'n=100, cens=0%'),
    (15, 0.2, 'n=75 , cens=20%'),
    (15, 0.4, 'n=75 , cens=40%'),
]

for (m,cr,label) in scenarios:
    rng  = np.random.default_rng(2024)
    ests = []
    for rep in range(R):
        Yr,Sr,Cr = simulate(m,cr,rng)
        try:
            d_init = np.sort(rng.uniform(0.25,0.58,2))
            res = em_rfl(Yr,Sr,2,d_init,cens=Cr,max_iter=300,tol=1e-6)
            se  = louis_se(Yr,Sr,Cr,res)
            ests.append([res['b0'],res['b1'],res['sig'],se[0],se[1],se[2]])
        except:
            pass
    if not ests: continue
    ests = np.array(ests)
    for j,name in enumerate(NAMES):
        ev,sv   = ests[:,j], ests[:,j+3]
        bias    = np.nanmean(ev - TRUE_VALS[j])
        rmse    = np.sqrt(np.nanmean((ev-TRUE_VALS[j])**2))
        lo,hi   = ev-1.96*sv, ev+1.96*sv
        cvg     = np.nanmean((TRUE_VALS[j]>=lo)&(TRUE_VALS[j]<=hi))
        se_avg  = np.nanmean(sv)
        prefix  = label if j==0 else ' '*24
        print(f"{prefix} | {name:<6} | {bias:+8.4f} | {rmse:8.4f} | {cvg:7.3f} | {se_avg:7.4f}")
    print()

print("Done.")
