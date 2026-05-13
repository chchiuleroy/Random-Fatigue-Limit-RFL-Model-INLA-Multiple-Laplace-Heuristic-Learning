"""
Semi-parametric RFL — Profile Likelihood SE
  ell_p(psi) = max_G  ell(psi, G)
  Hessian via central finite differences -> SE

Comparison: Louis SE  vs  Profile SE  vs  Bootstrap SE
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings; warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# Shared helpers (same as rfl_sim2.py)
# ════════════════════════════════════════════════════════════════
def mu_mat(S, deltas, b0, b1):
    diff  = S[:,None] - deltas[None,:]
    valid = diff > 1e-8
    lnd   = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.)
    mu    = np.where(valid, b0 + b1*lnd, -1e30)
    return mu, valid, lnd

def e_step_full(Y, S, cens, pis, deltas, b0, b1, sig):
    mu, valid, _ = mu_mat(S, deltas, b0, b1)
    n, K = len(Y), len(pis)
    log_pi = np.log(pis[None,:])
    logw   = np.full((n,K), -1e30)
    obs    = ~cens
    if obs.any():
        logw[obs] = np.where(valid[obs],
                             log_pi + norm.logpdf(Y[obs,None], mu[obs], sig), -1e30)
    if cens.any():
        a = (Y[cens,None] - mu[cens]) / sig
        logw[cens] = np.where(valid[cens],
                              log_pi + np.log(norm.sf(a).clip(1e-300)), -1e30)
    rm  = logw.max(1, keepdims=True)
    e   = np.exp(logw - rm)
    return e / e.sum(1, keepdims=True)

def m_step_delta_k(k, Y, S, cens, tau_k, b0, b1, sig):
    S_min = S.min(); obs = ~cens
    def neg_q(dk):
        if dk >= S_min-1e-6: return 1e10
        diff = S-dk; ok = diff>1e-8
        if not ok.any(): return 1e10
        mu_k = b0 + b1*np.log(np.maximum(diff,1e-300))
        v = (tau_k*ok*norm.logpdf(Y,mu_k,sig))[obs].sum()
        if cens.any():
            a = (Y[cens]-mu_k[cens])/sig
            v += (tau_k[cens]*ok[cens]*np.log(norm.sf(a).clip(1e-300))).sum()
        return -v
    return minimize_scalar(neg_q, bounds=(0.01,S_min-0.001), method='bounded').x

def m_step_params(Y, S, cens, tau, deltas, b0_old, b1_old, sig_old):
    mu, valid, lnd = mu_mat(S, deltas, b0_old, b1_old)
    w = tau * valid.astype(float)
    Y_aug = np.zeros_like(tau)
    obs = ~cens
    Y_aug[obs] = Y[obs,None]
    if cens.any():
        a_c = (Y[cens,None]-mu[cens])/sig_old
        lam = norm.pdf(a_c)/norm.sf(a_c).clip(1e-300)
        Y_aug[cens] = mu[cens]+sig_old*lam
    W=w.sum(); S01=(w*lnd).sum(); S11=(w*lnd**2).sum()
    T0=(w*Y_aug).sum(); T1=(w*lnd*Y_aug).sum()
    A=np.array([[W,S01],[S01,S11]])
    try: b0,b1=np.linalg.solve(A,[T0,T1])
    except: b0,b1=b0_old,b1_old
    mu2,valid2,_=mu_mat(S,deltas,b0,b1); w2=tau*valid2.astype(float)
    ss=(w2[obs]*(Y[obs,None]-mu2[obs])**2).sum(); wt=w2[obs].sum()
    if cens.any():
        a_c2=(Y[cens,None]-mu2[cens])/sig_old
        lam2=norm.pdf(a_c2)/norm.sf(a_c2).clip(1e-300)
        ss+=(w2[cens]*sig_old**2*(1+a_c2*lam2)).sum(); wt+=w2[cens].sum()
    return b0, b1, max(np.sqrt(ss/max(wt,1e-8)),0.01)

def log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig):
    mu,valid,_=mu_mat(S,deltas,b0,b1)
    comp=np.where(valid,pis[None,:],0.)
    ll=0.
    obs=~cens
    if obs.any():
        ll+=np.log((comp[obs]*norm.pdf(Y[obs,None],mu[obs],sig)).sum(1).clip(1e-300)).sum()
    if cens.any():
        a=(Y[cens,None]-mu[cens])/sig
        ll+=np.log((comp[cens]*norm.sf(a)).sum(1).clip(1e-300)).sum()
    return ll

def em_full(Y, S, K, delta_init, cens=None, max_iter=500, tol=1e-7):
    if cens is None: cens=np.zeros(len(Y),bool)
    pis=np.ones(K)/K; deltas=np.sort(np.asarray(delta_init[:K],float))
    obs=~cens
    lnd0=np.log(np.maximum(S[obs]-deltas.mean(),1e-8))
    Xm=np.column_stack([np.ones(obs.sum()),lnd0])
    b=np.linalg.lstsq(Xm,Y[obs],rcond=None)[0]
    b0,b1=float(b[0]),float(b[1]); sig=max(float(np.std(Y[obs]-Xm@b)),0.1)
    ll_prev=-np.inf
    for it in range(max_iter):
        tau=e_step_full(Y,S,cens,pis,deltas,b0,b1,sig)
        pis=np.maximum(tau.mean(0),1e-8); pis/=pis.sum()
        for k in range(K): deltas[k]=m_step_delta_k(k,Y,S,cens,tau[:,k],b0,b1,sig)
        b0,b1,sig=m_step_params(Y,S,cens,tau,deltas,b0,b1,sig)
        ll=log_lik_obs(Y,S,cens,pis,deltas,b0,b1,sig)
        if abs(ll-ll_prev)<tol: break
        ll_prev=ll
    return dict(pis=pis,deltas=deltas,b0=b0,b1=b1,sig=sig,ll=ll,iters=it+1)

# ════════════════════════════════════════════════════════════════
# Profile Likelihood
# ════════════════════════════════════════════════════════════════
def em_G_only(Y, S, cens, b0, b1, sig, pis_init, deltas_init, max_iter=300, tol=1e-9):
    """Fix (b0,b1,sig), optimise G = {pi_k, Delta_k} only."""
    pis    = pis_init.copy()
    deltas = deltas_init.copy()
    K      = len(pis)
    ll_prev= -np.inf
    for _ in range(max_iter):
        tau    = e_step_full(Y, S, cens, pis, deltas, b0, b1, sig)
        pis    = np.maximum(tau.mean(0), 1e-8); pis /= pis.sum()
        for k in range(K):
            deltas[k] = m_step_delta_k(k, Y, S, cens, tau[:,k], b0, b1, sig)
        ll     = log_lik_obs(Y, S, cens, pis, deltas, b0, b1, sig)
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    return pis, deltas, ll

def profile_ll(Y, S, cens, b0, b1, sig, pis0, deltas0):
    """ℓ_p(ψ) = max_G ℓ(ψ,G), warm-started from (pis0, deltas0)."""
    if sig <= 0: return -1e30
    _, _, ll = em_G_only(Y, S, cens, b0, b1, sig, pis0, deltas0)
    return ll

def profile_hessian_se(Y, S, cens, res, h=None):
    """
    -∂²ℓ_p/∂ψ² via central finite differences.
    Returns SE for (b0, b1, sig).
    """
    b0, b1, sig = res['b0'], res['b1'], res['sig']
    pis0, deltas0 = res['pis'], res['deltas']

    if h is None:
        h = np.array([0.05, 0.08, 0.02])   # tuned for typical RFL scale

    def pl(db0=0., db1=0., ds=0.):
        return profile_ll(Y, S, cens, b0+db0, b1+db1, sig+ds, pis0, deltas0)

    ll0 = pl()
    e   = [np.array([h[0],0,0]), np.array([0,h[1],0]), np.array([0,0,h[2]])]

    # build 3×3 numerical Hessian of ℓ_p
    H = np.zeros((3,3))
    for i in range(3):
        # diagonal: central second difference
        dp, dm = dict(zip(['db0','db1','ds'], e[i])), dict(zip(['db0','db1','ds'], -e[i]))
        H[i,i] = (pl(**dp) - 2*ll0 + pl(**dm)) / h[i]**2
        for j in range(i+1, 3):
            # off-diagonal: 4-point cross difference
            pp=dict(zip(['db0','db1','ds'], e[i]+e[j]))
            pm=dict(zip(['db0','db1','ds'], e[i]-e[j]))
            mp=dict(zip(['db0','db1','ds'],-e[i]+e[j]))
            mm=dict(zip(['db0','db1','ds'],-e[i]-e[j]))
            H[i,j]=H[j,i]=(pl(**pp)-pl(**pm)-pl(**mp)+pl(**mm))/(4*h[i]*h[j])

    try:
        cov = np.linalg.inv(-H)
        se  = np.sqrt(np.abs(np.diag(cov)))
        return se, cov, H
    except:
        return np.full(3,np.nan), None, H

def louis_se(Y, S, cens, res):
    pis,deltas,b0,b1,sig = res['pis'],res['deltas'],res['b0'],res['b1'],res['sig']
    mu,valid,lnd = mu_mat(S, deltas, b0, b1)
    tau = e_step_full(Y, S, cens, pis, deltas, b0, b1, sig)
    r   = (Y[:,None]-mu)/sig; w=tau*valid.astype(float)
    G   = np.stack([r/sig, r*lnd/sig, (r**2-1)/sig], axis=2)   # (n,K,3)
    I_c = np.array([
        [(w/sig**2).sum(),        (w*lnd/sig**2).sum(),      0.],
        [(w*lnd/sig**2).sum(),    (w*lnd**2/sig**2).sum(),   0.],
        [0., 0., (w*(3*r**2-1)/sig**2).sum()]
    ])
    E_ggT  = np.einsum('nkp,nkq,nk->pq', G, G, tau)
    s      = np.einsum('nkp,nk->np', G, tau)
    I_obs  = I_c - (E_ggT - s.T@s)
    try:
        return np.sqrt(np.abs(np.diag(np.linalg.inv(I_obs))))
    except:
        return np.full(3, np.nan)

# ════════════════════════════════════════════════════════════════
# REAL DATA — SE comparison
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
S_arr,Y_arr=[],[]
for s,ys in raw.items():
    for y in ys: S_arr.append(s); Y_arr.append(np.log(y))
S_r=np.array(S_arr); Y_r=np.array(Y_arr); cens_r=np.zeros(len(Y_r),bool)

print("="*60); print("REAL DATA  K=2  SE comparison"); print("="*60)
best_ll=-np.inf; best_r=None
for seed in range(15):
    np.random.seed(seed)
    d=np.sort(np.random.uniform(0.30,0.62,2))
    try:
        r=em_full(Y_r,S_r,2,d,max_iter=2000,tol=1e-9)
        if r['ll']>best_ll: best_ll=r['ll']; best_r=r
    except: pass

r=best_r
print(f"b0={r['b0']:.5f}  b1={r['b1']:.5f}  sigma={r['sig']:.5f}")
print(f"pi={r['pis'].round(5)}  D={r['deltas'].round(6)}")
print(f"ll={r['ll']:.5f}  iters={r['iters']}")
print()

se_louis   = louis_se(Y_r, S_r, cens_r, r)
se_prof, cov_prof, H_prof = profile_hessian_se(Y_r, S_r, cens_r, r)

print(f"{'':12} {'SE_Louis':>10} {'SE_Profile':>12} {'ratio':>8}")
for j, name in enumerate(['b0','b1','sigma']):
    ratio = se_prof[j]/se_louis[j] if se_louis[j]>0 else np.nan
    print(f"  {name:<10} {se_louis[j]:>10.4f} {se_prof[j]:>12.4f} {ratio:>8.3f}x")

print()
print("95% CI  (Profile SE):")
for j, (name, val) in enumerate(zip(['b0','b1','sigma'],[r['b0'],r['b1'],r['sig']])):
    print(f"  {name}: [{val-1.96*se_prof[j]:.4f}, {val+1.96*se_prof[j]:.4f}]")

print()
print("Profile Hessian:")
print(np.round(H_prof,4))
print("Profile covariance:")
if cov_prof is not None: print(np.round(cov_prof,4))

# ════════════════════════════════════════════════════════════════
# SIMULATION — Profile SE coverage
# ════════════════════════════════════════════════════════════════
TRUE=dict(b0=-9.1,b1=-8.0,sig=0.60,pis=np.array([0.7,0.3]),deltas=np.array([0.40,0.55]))
STRESS=np.array([0.675,0.75,0.825,0.9,0.95])
TRUE_VALS=np.array([TRUE['b0'],TRUE['b1'],TRUE['sig']])
NAMES=['b0','b1','sigma']
R=200

def simulate(m, cr, rng):
    b0,b1,sig=TRUE['b0'],TRUE['b1'],TRUE['sig']
    pis,deltas=TRUE['pis'],TRUE['deltas']
    Y_,S_,C_=[],[],[]
    for s in STRESS:
        dk=rng.choice(deltas,size=m,p=pis)
        mu=b0+b1*np.log(np.maximum(s-dk,1e-8))
        y=rng.normal(mu,sig)
        ct=np.quantile(y,1-cr) if cr>0 else np.inf
        csd=y>ct
        Y_.extend(np.where(csd,ct,y)); S_.extend([s]*m); C_.extend(csd)
    return np.array(Y_),np.array(S_),np.array(C_,bool)

print(); print("="*60)
print(f"SIMULATION  R={R}  True K=2  delta=(0.40,0.55)")
print("="*60)
print(f"{'Scenario':<22} | {'Param':<5} | {'Bias':>7} | {'RMSE':>7} | {'Cvg_Louis':>10} | {'Cvg_Profile':>12}")
print("-"*76)

scenarios=[(10,0.0,'n=50 , 0%'),(15,0.0,'n=75 , 0%'),(20,0.0,'n=100, 0%'),
           (15,0.2,'n=75 ,20%'),(15,0.4,'n=75 ,40%')]

for (m,cr,label) in scenarios:
    rng=np.random.default_rng(2025)
    ests=[]
    for rep in range(R):
        Yr,Sr,Cr=simulate(m,cr,rng)
        try:
            d_init=np.sort(rng.uniform(0.25,0.58,2))
            res=em_full(Yr,Sr,2,d_init,cens=Cr,max_iter=300,tol=1e-6)
            sl=louis_se(Yr,Sr,Cr,res)
            sp,_,_=profile_hessian_se(Yr,Sr,Cr,res)
            ests.append([res['b0'],res['b1'],res['sig'],sl[0],sl[1],sl[2],sp[0],sp[1],sp[2]])
        except: pass
    if not ests: continue
    ests=np.array(ests)
    for j,name in enumerate(NAMES):
        ev=ests[:,j]; sl_j=ests[:,j+3]; sp_j=ests[:,j+6]
        bias=np.nanmean(ev-TRUE_VALS[j]); rmse=np.sqrt(np.nanmean((ev-TRUE_VALS[j])**2))
        cvg_l=np.nanmean((TRUE_VALS[j]>=ev-1.96*sl_j)&(TRUE_VALS[j]<=ev+1.96*sl_j))
        cvg_p=np.nanmean((TRUE_VALS[j]>=ev-1.96*sp_j)&(TRUE_VALS[j]<=ev+1.96*sp_j))
        prefix=label if j==0 else ' '*22
        print(f"{prefix} | {name:<5} | {bias:+7.4f} | {rmse:7.4f} | {cvg_l:>10.3f} | {cvg_p:>12.3f}")
    print()

print("Done.")
