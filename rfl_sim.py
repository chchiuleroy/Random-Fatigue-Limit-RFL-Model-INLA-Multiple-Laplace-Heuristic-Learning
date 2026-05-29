"""
Semi-parametric RFL Model
Step 1: K=2 convergence check (max_iter=2000)
Step 2: Simulation study — Bias / RMSE / Coverage (Louis SE)
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings; warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# Core EM (vectorized)
# ════════════════════════════════════════════════════════════════
def mu_mat(S, deltas, b0, b1):
    diff  = S[:,None] - deltas[None,:]
    valid = diff > 1e-8
    lnd   = np.where(valid, np.log(np.maximum(diff,1e-300)), 0.)
    mu    = np.where(valid, b0 + b1*lnd, -1e30)
    return mu, valid, lnd

def e_step(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    logw = np.where(valid,
                    np.log(pis[None,:]) + norm.logpdf(Y[:,None], mu, sig),
                    -1e30)
    rm   = logw.max(1, keepdims=True)
    tau  = np.exp(logw - rm - np.log(np.exp(logw-rm).sum(1, keepdims=True)))
    return tau

def m_step(Y, S, tau, deltas_prev, b0_prev, b1_prev, sig_prev):
    n, K = tau.shape
    S_min = S.min()

    # ── delta_k (Brent 1D) ──────────────────────────────────
    deltas = deltas_prev.copy()
    for k in range(K):
        tk = tau[:,k]
        def neg_q(dk, tk=tk):
            if dk >= S_min-1e-6: return 1e10
            diff = S - dk
            ok   = diff > 1e-8
            if not ok.any(): return 1e10
            mu   = b0_prev + b1_prev*np.log(np.maximum(diff,1e-300))
            return -(tk * ok * norm.logpdf(Y, mu, sig_prev)).sum()
        res = minimize_scalar(neg_q, bounds=(0.01, S_min-0.001), method='bounded')
        deltas[k] = res.x

    # ── b0, b1 (weighted OLS normal equations) ──────────────
    mu, valid, lnd = mu_mat(S, deltas, b0_prev, b1_prev)
    w  = tau * valid.astype(float)          # (n,K)
    W  = w.sum()
    S01 = (w * lnd).sum()
    S11 = (w * lnd**2).sum()
    T0  = (w * Y[:,None]).sum()
    T1  = (w * lnd * Y[:,None]).sum()
    A   = np.array([[W, S01],[S01, S11]])
    try:
        b0, b1 = np.linalg.solve(A, [T0, T1])
    except np.linalg.LinAlgError:
        b0, b1 = b0_prev, b1_prev

    # ── sigma ────────────────────────────────────────────────
    mu2, valid2, _ = mu_mat(S, deltas, b0, b1)
    w2  = tau * valid2.astype(float)
    sig = max(np.sqrt((w2*(Y[:,None]-mu2)**2).sum() / w2.sum()), 0.01)

    return deltas, b0, b1, sig

def log_lik(Y, S, pis, deltas, b0, b1, sig):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    comp = np.where(valid, pis[None,:]*norm.pdf(Y[:,None], mu, sig), 0.)
    return np.log(comp.sum(1).clip(1e-300)).sum()

def em_rfl(Y, S, K, delta_init, max_iter=2000, tol=1e-8):
    pis    = np.ones(K)/K
    deltas = np.sort(np.asarray(delta_init[:K], dtype=float))
    lnd0   = np.log(np.maximum(S - deltas.mean(), 1e-8))
    Xm     = np.column_stack([np.ones(len(Y)), lnd0])
    b      = np.linalg.lstsq(Xm, Y, rcond=None)[0]
    b0, b1 = float(b[0]), float(b[1])
    sig    = max(float(np.std(Y - Xm@b)), 0.1)

    ll_prev = -np.inf
    for it in range(max_iter):
        tau    = e_step(Y, S, pis, deltas, b0, b1, sig)
        pis    = np.maximum(tau.mean(0), 1e-8); pis /= pis.sum()
        deltas, b0, b1, sig = m_step(Y, S, tau, deltas, b0, b1, sig)
        ll     = log_lik(Y, S, pis, deltas, b0, b1, sig)
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll

    return dict(pis=pis, deltas=deltas, b0=b0, b1=b1, sig=sig, ll=ll, iters=it+1)

# ── Louis (1982) observed information ────────────────────────
def louis_se(Y, S, res):
    """Return SE for (b0, b1, sig) via Louis formula"""
    pis, deltas, b0, b1, sig = res['pis'], res['deltas'], res['b0'], res['b1'], res['sig']
    K = len(pis)
    tau = e_step(Y, S, pis, deltas, b0, b1, sig)   # (n,K)
    mu, valid, lnd = mu_mat(S, deltas, b0, b1)

    # score contributions s_i = (ds_b0, ds_b1, ds_sig) for each i
    # d log f_k / d b0  = (y-mu)/sig^2
    # d log f_k / d b1  = (y-mu)/sig^2 * lnd
    # d log f_k / d sig = (y-mu)^2/sig^3 - 1/sig
    r = (Y[:,None] - mu) / sig                      # (n,K) standardised resid
    w = tau * valid.astype(float)

    s_b0  = (w * r / sig).sum(1)                    # (n,)
    s_b1  = (w * r * lnd / sig).sum(1)
    s_sig = (w * (r**2 - 1) / sig).sum(1)
    scores = np.column_stack([s_b0, s_b1, s_sig])   # (n,3)

    # complete-data Hessian (expected, negated) — diagonal blocks
    H_b0b0  = (w / sig**2).sum()
    H_b1b1  = (w * lnd**2 / sig**2).sum()
    H_b0b1  = (w * lnd / sig**2).sum()
    H_ss    = (w * (3*r**2 - 1) / sig**2).sum()     # approx
    I_c = np.array([
        [H_b0b0, H_b0b1, 0.],
        [H_b0b1, H_b1b1, 0.],
        [0.,     0.,     H_ss]
    ])

    # missing information = Var[score | obs] = sum_i s_i s_i^T  (Louis)
    I_mis = scores.T @ scores

    I_obs = I_c - I_mis
    try:
        cov   = np.linalg.inv(I_obs)
        se    = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    return se   # [se_b0, se_b1, se_sig]

# ════════════════════════════════════════════════════════════════
# Step 1 — Real data, K=2 with max_iter=2000
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
S_real = np.array(S_arr); Y_real = np.array(Y_arr); n_real = len(Y_real)

print("=" * 60)
print("STEP 1 — Real data K=2  (max_iter=2000, tol=1e-8)")
print("=" * 60)
best_ll = -np.inf; best_res = None
for seed in range(15):
    np.random.seed(seed)
    d_init = np.sort(np.random.uniform(0.30, 0.62, 2))
    try:
        r = em_rfl(Y_real, S_real, 2, d_init)
        if r['ll'] > best_ll: best_ll = r['ll']; best_res = r
    except: pass

r = best_res
p2 = 3 + 2*2 - 1
print(f"Converged at iter = {r['iters']}  (tol=1e-8)")
print(f"log-lik = {r['ll']:.6f}   BIC = {-2*r['ll']+p2*np.log(n_real):.4f}")
print(f"b0  = {r['b0']:.5f}   b1  = {r['b1']:.5f}   sigma = {r['sig']:.5f}")
print(f"pi  = {r['pis'].round(5)}")
print(f"D   = {r['deltas'].round(6)}")
se = louis_se(Y_real, S_real, r)
print(f"SE(Louis):  se_b0={se[0]:.4f}  se_b1={se[1]:.4f}  se_sig={se[2]:.4f}")
print(f"95% CI b0: [{r['b0']-1.96*se[0]:.4f}, {r['b0']+1.96*se[0]:.4f}]")
print(f"95% CI b1: [{r['b1']-1.96*se[1]:.4f}, {r['b1']+1.96*se[1]:.4f}]")
print(f"95% CI s : [{r['sig']-1.96*se[2]:.4f}, {r['sig']+1.96*se[2]:.4f}]")

# ════════════════════════════════════════════════════════════════
# Step 2 — Simulation study
# TRUE: K=2, b0=-9.1, b1=-8.0, sig=0.60, pi=(0.7,0.3), D=(0.50,0.56)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 2 — Simulation Study (Type I censoring)")
print("=" * 60)

TRUE = dict(b0=-9.1, b1=-8.0, sig=0.60,
            pis=np.array([0.7,0.3]), deltas=np.array([0.50,0.56]))
STRESS = np.array([0.675, 0.75, 0.825, 0.9, 0.95])
R = 200   # replications (quick run; set to 500 for final)

def simulate_data(m, cens_rate, rng):
    """m obs per stress level; Type-I censoring at quantile (1-cens_rate)"""
    b0,b1,sig = TRUE['b0'], TRUE['b1'], TRUE['sig']
    pis, deltas = TRUE['pis'], TRUE['deltas']
    Y_all, S_all, C_all, delta_true = [], [], [], []
    for s in STRESS:
        # draw Delta from discrete g
        dk = rng.choice(deltas, size=m, p=pis)
        diff = s - dk
        mu   = b0 + b1*np.log(np.maximum(diff,1e-8))
        y    = rng.normal(mu, sig)
        # Type-I censoring time = (1-cens_rate) quantile of uncensored
        cens_t = np.quantile(y, 1.0-cens_rate) if cens_rate > 0 else np.inf
        censored = (y > cens_t)
        Y_all.extend(np.where(censored, cens_t, y))
        S_all.extend([s]*m)
        C_all.extend(censored.tolist())
        delta_true.extend(dk)
    return (np.array(Y_all), np.array(S_all),
            np.array(C_all, dtype=bool))

def em_rfl_censored(Y, S, cens, K, delta_init, max_iter=500, tol=1e-7):
    """EM with Type-I right censoring"""
    pis    = np.ones(K)/K
    deltas = np.sort(np.asarray(delta_init[:K], float))
    obs    = ~cens
    # init on uncensored
    lnd0   = np.log(np.maximum(S[obs] - deltas.mean(), 1e-8))
    Xm     = np.column_stack([np.ones(obs.sum()), lnd0])
    b      = np.linalg.lstsq(Xm, Y[obs], rcond=None)[0]
    b0, b1 = float(b[0]), float(b[1])
    sig    = max(float(np.std(Y[obs] - Xm@b)), 0.1)
    n      = len(Y)
    ll_prev = -np.inf

    for it in range(max_iter):
        mu, valid, lnd = mu_mat(S, deltas, b0, b1)
        # E-step (handle censored separately)
        logw_obs  = np.where(valid[obs],
                             np.log(pis[None,:])+norm.logpdf(Y[obs,None],mu[obs],sig),
                             -1e30)
        logw_cens = np.where(valid[~obs],
                             np.log(pis[None,:])+norm.logcdf(-((Y[~obs,None]-mu[~obs])/sig)),
                             # log P(Y>c) = log Phi^c((c-mu)/sig)
                             -1e30)
        # rewrite: log S(c) = log Phi^c(a) = log(1-Phi(a))
        if (~obs).any():
            a_cens = (Y[~obs,None] - mu[~obs]) / sig
            logw_cens = np.where(valid[~obs],
                                 np.log(pis[None,:]) + np.log(norm.sf(a_cens).clip(1e-300)),
                                 -1e30)
        tau = np.zeros((n, K))
        def softmax_rows(lw):
            rm = lw.max(1,keepdims=True)
            e  = np.exp(lw-rm)
            return e/e.sum(1,keepdims=True)
        tau[obs]  = softmax_rows(logw_obs)
        if (~obs).any():
            tau[~obs] = softmax_rows(logw_cens)

        # M: pi
        pis = np.maximum(tau.mean(0),1e-8); pis /= pis.sum()

        # M: delta_k
        S_min = S.min()
        for k in range(K):
            tk = tau[:,k]
            def neg_q(dk, tk=tk, obs=obs):
                if dk>=S_min-1e-6: return 1e10
                diff=S-dk; ok=diff>1e-8
                if not ok.any(): return 1e10
                mu_k=b0+b1*np.log(np.maximum(diff,1e-300))
                v  = (tk*ok*norm.logpdf(Y,mu_k,sig))[obs].sum()
                if (~obs).any():
                    a_c=(Y[~obs]-mu_k[~obs])/sig
                    v += (tk[~obs]*ok[~obs]*np.log(norm.sf(a_c).clip(1e-300))).sum()
                return -v
            res = minimize_scalar(neg_q, bounds=(0.01,S_min-0.001), method='bounded')
            deltas[k] = res.x

        # M: b0,b1 using only uncensored (simple version)
        mu2, valid2, lnd2 = mu_mat(S[obs], deltas, b0, b1)
        w2 = tau[obs] * valid2.astype(float)
        W=w2.sum(); S01=(w2*lnd2).sum(); S11=(w2*lnd2**2).sum()
        T0=(w2*Y[obs,None]).sum(); T1=(w2*lnd2*Y[obs,None]).sum()
        A=np.array([[W,S01],[S01,S11]])
        try: b0,b1=np.linalg.solve(A,[T0,T1])
        except: pass

        # M: sigma (augment censored with E[Y|Y>c])
        mu3,valid3,_=mu_mat(S,deltas,b0,b1)
        w3=tau*valid3.astype(float)
        # uncensored
        ss = (w3[obs]*(Y[obs,None]-mu3[obs])**2).sum()
        wt = w3[obs].sum()
        # censored: E[(Y-mu)^2|Y>c] = sig^2*(1 + a*lambda) where lambda=phi(a)/Phi^c(a)
        if (~obs).any():
            a_c=(Y[~obs,None]-mu3[~obs])/sig
            lam=norm.pdf(a_c)/norm.sf(a_c).clip(1e-300)
            ss += (w3[~obs]*sig**2*(1+a_c*lam)).sum()
            wt += w3[~obs].sum()
        sig = max(np.sqrt(ss/max(wt,1e-8)),0.01)

        # log-lik observed
        comp_o=np.where(valid[obs],  pis[None,:]*norm.pdf(Y[obs,None], mu[obs],sig),0.)
        comp_c=np.where(valid[~obs], pis[None,:]*norm.sf((Y[~obs,None]-mu[~obs])/sig),0.) if (~obs).any() else np.zeros((0,K))
        ll = (np.log(comp_o.sum(1).clip(1e-300)).sum() +
              (np.log(comp_c.sum(1).clip(1e-300)).sum() if (~obs).any() else 0.))
        if abs(ll-ll_prev)<tol: break
        ll_prev=ll

    return dict(pis=pis,deltas=deltas,b0=b0,b1=b1,sig=sig,ll=ll,iters=it+1)

# ── Run simulation ────────────────────────────────────────────
scenarios = [
    (10, 0.0, 'n=50  , cens=0%'),
    (15, 0.0, 'n=75  , cens=0%'),
    (20, 0.0, 'n=100 , cens=0%'),
    (15, 0.2, 'n=75  , cens=20%'),
    (15, 0.4, 'n=75  , cens=40%'),
]

TRUE_VALS = np.array([TRUE['b0'], TRUE['b1'], TRUE['sig']])
NAMES     = ['b0','b1','sigma']

print(f"True: b0={TRUE['b0']}, b1={TRUE['b1']}, sigma={TRUE['sig']}")
print(f"True G: pi=({TRUE['pis'][0]},{TRUE['pis'][1]}), Delta=({TRUE['deltas'][0]},{TRUE['deltas'][1]})")
print(f"R = {R} replications per scenario\n")
print(f"{'Scenario':<22} | {'Param':<6} | {'Bias':>8} | {'RMSE':>8} | {'Cvg95':>7}")
print("-"*65)

for (m, cr, label) in scenarios:
    rng   = np.random.default_rng(42)
    ests  = []
    for rep in range(R):
        Yr, Sr, Cr = simulate_data(m, cr, rng)
        try:
            d_init = np.sort(rng.uniform(0.3,0.58,2))
            if cr == 0.0:
                res = em_rfl(Yr, Sr, 2, d_init, max_iter=300, tol=1e-6)
            else:
                res = em_rfl_censored(Yr, Sr, Cr, 2, d_init, max_iter=300, tol=1e-6)
            # Louis SE
            try:
                se = louis_se(Yr, Sr, res)
            except:
                se = np.full(3, np.nan)
            ests.append([res['b0'],res['b1'],res['sig'],se[0],se[1],se[2]])
        except:
            pass
    ests = np.array(ests)
    if len(ests) == 0: continue
    for j, name in enumerate(NAMES):
        est_j = ests[:,j]; se_j = ests[:,j+3]
        bias  = np.nanmean(est_j - TRUE_VALS[j])
        rmse  = np.sqrt(np.nanmean((est_j - TRUE_VALS[j])**2))
        lo    = est_j - 1.96*se_j; hi = est_j + 1.96*se_j
        cvg   = np.nanmean((TRUE_VALS[j] >= lo) & (TRUE_VALS[j] <= hi))
        prefix = label if j==0 else ' '*22
        print(f"{prefix} | {name:<6} | {bias:+8.4f} | {rmse:8.4f} | {cvg:7.3f}")
    print()

print("Done.")
