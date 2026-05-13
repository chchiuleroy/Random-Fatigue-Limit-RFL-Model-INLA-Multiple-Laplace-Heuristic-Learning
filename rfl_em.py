"""
Semi-parametric Random Fatigue-Limit Model
  f(y|Delta) = Normal(b0 + b1*log(S-Delta), sigma^2)
  g(Delta)   = NPMLE discrete: sum_k pi_k * delta(Delta_k)
EM algorithm (vectorized) + BIC selection
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings; warnings.filterwarnings('ignore')

# ── 資料 ─────────────────────────────────────────────────────────────────────
raw = {
    0.675: [102.95,280.32,339.83,366.9,485.62,658.96,896.33,
            1241.76,1250.2,1329.78,1399.83,1459.14,3249.82,11748.1,11748.1],
    0.75:  [6.71,9.93,12.6,15.58,16.19,17.28,18.62,
            20.3,24.9,26.26,27.94,36.35,48.42,50.09,67.34],
    0.825: [1.246,1.258,1.46,1.492,2.4,2.41,2.59,
            2.903,3.33,3.59,3.847,4.11,4.82,5.56,5.598],
    0.9:   [0.201,0.216,0.226,0.252,0.257,0.295,0.311,
            0.342,0.356,0.451,0.457,0.509,0.54,0.68,1.129],
    0.95:  [0.037,0.072,0.074,0.076,0.083,0.085,0.105,
            0.109,0.12,0.123,0.143,0.203,0.206,0.217,0.257]
}
S_arr, Y_arr = [], []
for s, ys in raw.items():
    for y in ys:
        S_arr.append(s); Y_arr.append(np.log(y))
S = np.array(S_arr); Y = np.array(Y_arr); n = len(Y)

# ── 核心函數（全向量化）─────────────────────────────────────────────────────
def mu_matrix(S, deltas, b0, b1):
    """Return (n, K) matrix of mu_ik = b0 + b1*log(S_i - Delta_k)"""
    # S: (n,), deltas: (K,)
    diff = S[:,None] - deltas[None,:]          # (n, K)
    valid = diff > 1e-8
    log_diff = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.0)
    mu = np.where(valid, b0 + b1 * log_diff, -1e30)
    return mu, valid                            # (n,K), (n,K)

def e_step(Y, S, pis, deltas, b0, b1, sig):
    """Vectorized E-step: returns tau (n,K)"""
    mu, valid = mu_matrix(S, deltas, b0, b1)   # (n,K)
    # log pdf
    log_tau = np.where(valid,
                       np.log(pis[None,:]) + norm.logpdf(Y[:,None], mu, sig),
                       -1e30)
    # log-sum-exp per row
    row_max = log_tau.max(axis=1, keepdims=True)
    log_tau_shifted = log_tau - row_max
    log_row_sum = np.log(np.exp(log_tau_shifted).sum(axis=1, keepdims=True)) + row_max
    tau = np.exp(log_tau - log_row_sum)
    return tau

def m_step_params(Y, S, tau, deltas, b0_prev, b1_prev, sig_prev):
    """Update b0, b1, sigma via weighted OLS (vectorized)"""
    mu, valid = mu_matrix(S, deltas, b0_prev, b1_prev)
    diff = S[:,None] - deltas[None,:]
    log_diff = np.where(valid, np.log(np.maximum(diff, 1e-300)), 0.0)

    # Weighted OLS: design matrix cols = [1, log(S_i - Delta_k)] for each (i,k) pair
    # weight = tau_ik, response = Y_i
    w = tau * valid.astype(float)               # (n,K)
    W_sum = w.sum()

    # X = [1, log_diff], shape tricks for normal equations
    # sum_ik w_ik * 1       = W_sum
    # sum_ik w_ik * x_ik    = (w * log_diff).sum()
    # sum_ik w_ik * x_ik^2  = (w * log_diff^2).sum()
    # sum_ik w_ik * Y_i     = (w * Y[:,None]).sum()
    # sum_ik w_ik * x_ik*Y_i= (w * log_diff * Y[:,None]).sum()
    S00 = W_sum
    S01 = (w * log_diff).sum()
    S11 = (w * log_diff**2).sum()
    T0  = (w * Y[:,None]).sum()
    T1  = (w * log_diff * Y[:,None]).sum()

    A = np.array([[S00, S01],[S01, S11]])
    b_vec = np.array([T0, T1])
    try:
        b0_new, b1_new = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        b0_new, b1_new = b0_prev, b1_prev

    # update sigma
    mu_new = np.where(valid, b0_new + b1_new * log_diff, 0.0)
    resid2 = (Y[:,None] - mu_new)**2
    sig2 = (w * resid2).sum() / W_sum
    sig_new = max(np.sqrt(sig2), 0.01)

    return b0_new, b1_new, sig_new

def m_step_delta_k(k, Y, S, tau_k, b0, b1, sig):
    """1D Brent optimization for Delta_k"""
    S_min = S.min()
    def neg_q(dk):
        if dk >= S_min - 1e-6: return 1e10
        diff = S - dk
        valid = diff > 1e-8
        if not valid.any(): return 1e10
        mu = b0 + b1 * np.log(np.maximum(diff, 1e-300))
        val = (tau_k * valid * norm.logpdf(Y, mu, sig)).sum()
        return -val
    res = minimize_scalar(neg_q, bounds=(0.01, S_min - 0.001), method='bounded')
    return res.x

def log_likelihood(Y, S, pis, deltas, b0, b1, sig):
    """Observed-data log-likelihood (vectorized)"""
    mu, valid = mu_matrix(S, deltas, b0, b1)
    comp = np.where(valid, pis[None,:] * norm.pdf(Y[:,None], mu, sig), 0.0)
    return np.log(comp.sum(axis=1).clip(1e-300)).sum()

# ── EM 主程式 ────────────────────────────────────────────────────────────────
def em_rfl(Y, S, K, delta_init, max_iter=300, tol=1e-6):
    pis    = np.ones(K) / K
    deltas = np.sort(np.array(delta_init[:K]))

    # OLS init
    x0 = np.log(np.maximum(S - deltas.mean(), 1e-8))
    Xm = np.column_stack([np.ones(n), x0])
    b  = np.linalg.lstsq(Xm, Y, rcond=None)[0]
    b0, b1 = b[0], b[1]
    sig = max(np.std(Y - Xm.dot(b)), 0.1)

    ll_prev = -np.inf
    for it in range(max_iter):
        # E
        tau = e_step(Y, S, pis, deltas, b0, b1, sig)

        # M: pi
        pis = tau.mean(axis=0)
        pis = np.maximum(pis, 1e-8); pis /= pis.sum()

        # M: delta_k
        for k in range(K):
            deltas[k] = m_step_delta_k(k, Y, S, tau[:,k], b0, b1, sig)

        # M: b0, b1, sigma
        b0, b1, sig = m_step_params(Y, S, tau, deltas, b0, b1, sig)

        # convergence
        ll = log_likelihood(Y, S, pis, deltas, b0, b1, sig)
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll

    return dict(pis=pis, deltas=deltas, b0=b0, b1=b1, sig=sig, ll=ll, iters=it+1)

# ── BIC 選 K ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Semi-parametric RFL Model  |  Real Fatigue Data")
print("=" * 60)
print(f"n = {n},  stress levels = {sorted(set(S_arr))}")
print()

best_bic = np.inf; best_K = 1; all_res = {}
for K in [1, 2, 3, 4]:
    best_ll = -np.inf; best_res = None
    for seed in range(8):
        np.random.seed(seed)
        d_init = np.sort(np.random.uniform(0.30, 0.62, K))
        try:
            res = em_rfl(Y, S, K, d_init)
            if res['ll'] > best_ll:
                best_ll = res['ll']; best_res = res
        except Exception as e:
            pass
    all_res[K] = best_res
    p_K  = 3 + 2*K - 1       # b0, b1, sig, (K-1 free pi), K deltas
    bic  = -2*best_res['ll'] + p_K * np.log(n)
    aic  = -2*best_res['ll'] + 2*p_K
    flag = " <-- BIC best" if bic < best_bic else ""
    if bic < best_bic: best_bic = bic; best_K = K
    print(f"K={K}:  ll={best_res['ll']:9.4f}  #par={p_K}  BIC={bic:8.3f}  AIC={aic:8.3f}{flag}")
    print(f"      b0={best_res['b0']:.4f}  b1={best_res['b1']:.4f}  sigma={best_res['sig']:.4f}  iters={best_res['iters']}")
    pi_str = "  ".join([f"pi{k+1}={best_res['pis'][k]:.4f}" for k in range(K)])
    d_str  = "  ".join([f"D{k+1}={best_res['deltas'][k]:.5f}" for k in range(K)])
    print(f"      {pi_str}")
    print(f"      {d_str}")
    print()

print(f"=> BIC selects K = {best_K}")
