"""
rfl_mcem.py -- MCEM for RFL: SEV conditional + LogNormal g(Delta)
E-step: rejection sampling from P(Delta_i | y_i, S_i)
M-step: (b0,b1,sig) via BFGS; (mu_d, sig_d) via closed-form LogNormal MLE
Compare ASSE with INLA (Normal+GH, ASSE=8.63) and NPMLE methods
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from scipy.stats import lognorm, norm
from scipy.optimize import minimize, minimize_scalar
import warnings; warnings.filterwarnings('ignore')

# ── SEV functions ─────────────────────────────────────────────────
def sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def sev_logsf(y, mu, sig):
    return -np.exp((y - mu) / sig)

def mu_fn(s, d, b0, b1):
    return b0 + b1 * np.log(np.maximum(s - d, 1e-8))

# ── E-step: sample Delta_i from posterior P(Delta|y_i, S_i) ──────
def sample_delta_posterior(y, s, b0, b1, sig, mu_d, sig_d, M, rng):
    """
    Rejection sampling with LogNormal prior as proposal.
    Fallback to grid evaluation if acceptance rate is too low.
    """
    # normalizing constant: max of SEV log-likelihood over Delta
    def neg_log_fsev(d):
        if d <= 0 or d >= s: return 1e10
        return -(mu_fn(s, d, b0, b1) - np.exp((y - mu_fn(s, d, b0, b1)) / sig))
        # above is wrong — compute properly:
    def neg_log_fsev2(d):
        if d <= 0 or d >= s: return 1e10
        mu = mu_fn(s, d, b0, b1)
        w  = (y - mu) / sig
        return -(w - np.exp(w))   # log f_SEV kernel (drops constant -log sig)

    res = minimize_scalar(neg_log_fsev2, bounds=(1e-6, s - 1e-4), method='bounded')
    log_fsev_max = -res.fun

    samples = []
    n_try   = 0
    while len(samples) < M and n_try < M * 500:
        n_try += 1
        d = lognorm.rvs(sig_d, scale=np.exp(mu_d), random_state=rng)
        if d <= 0 or d >= s:
            continue
        mu  = mu_fn(s, d, b0, b1)
        w   = (y - mu) / sig
        log_fsev = w - np.exp(w)
        if np.log(rng.uniform()) < log_fsev - log_fsev_max:
            samples.append(d)

    if len(samples) < M // 2:
        # grid fallback (analogous to INLA's GH)
        d_grid = np.linspace(1e-4, s - 1e-4, 800)
        log_h  = np.array([
            sev_logpdf(y, mu_fn(s, d, b0, b1), sig) +
            lognorm.logpdf(d, sig_d, scale=np.exp(mu_d))
            for d in d_grid
        ])
        log_h -= log_h.max()
        h = np.exp(log_h); h /= h.sum()
        idx = rng.choice(len(d_grid), size=M, p=h)
        return d_grid[idx]

    return np.array(samples[:M])

# ── M-step ────────────────────────────────────────────────────────
def m_step(Y, S, cens, all_samples, b0_old, b1_old, sig_old):
    obs = ~cens
    # (a) LogNormal parameters: closed-form MLE on samples
    all_log_d = np.concatenate([np.log(np.maximum(s, 1e-8)) for s in all_samples])
    mu_d_new  = float(all_log_d.mean())
    sig_d_new = float(np.maximum(all_log_d.std(), 0.001))

    # (b) (b0, b1, sig): BFGS on Q approximated by MC samples
    n_total = sum(len(s) for s in all_samples)
    def neg_Q(params):
        b0, b1, log_sig = params
        sig = np.exp(log_sig)
        if sig < 0.01 or sig > 10: return 1e10
        q = 0.
        for i in range(len(Y)):
            for d in all_samples[i]:
                mu = mu_fn(S[i], d, b0, b1)
                if obs[i]:
                    q += sev_logpdf(Y[i], mu, sig)
                else:
                    q += sev_logsf(Y[i], mu, sig)
        return -q / n_total

    res = minimize(neg_Q, [b0_old, b1_old, np.log(sig_old)],
                   method='BFGS', options={'gtol':1e-5, 'maxiter':300})
    b0, b1, log_sig = res.x
    return b0, b1, float(np.exp(log_sig)), mu_d_new, sig_d_new

# ── MC log-likelihood estimate ────────────────────────────────────
def mc_loglik(Y, S, cens, b0, b1, sig, mu_d, sig_d, all_samples):
    """Importance-weighted log-likelihood using MC samples"""
    obs = ~cens; ll = 0.
    for i in range(len(Y)):
        vals = []
        for d in all_samples[i]:
            mu = mu_fn(S[i], d, b0, b1)
            lf = sev_logpdf(Y[i], mu, sig) if obs[i] else sev_logsf(Y[i], mu, sig)
            lg = lognorm.logpdf(d, sig_d, scale=np.exp(mu_d))
            lq = lognorm.logpdf(d, sig_d, scale=np.exp(mu_d))  # proposal = prior
            vals.append(lf)   # importance weight cancels (proposal = prior)
        ll += np.log(np.mean(np.exp(np.array(vals) - np.max(vals)))) + np.max(vals)
    return ll

EULER_GAMMA = 0.5772156649015328

# ── ASSE from posterior samples ───────────────────────────────────
def asse_from_samples(Y, S, b0, b1, sig, all_samples):
    # E[Y|Delta,mu,sig] = mu - sig*gamma for SEV, so correct for the shift
    fitted = np.array([
        np.mean([mu_fn(S[i], d, b0, b1) for d in all_samples[i]]) - sig * EULER_GAMMA
        for i in range(len(Y))
    ])
    ae = np.abs(Y - fitted)
    return ae.sum(), ae.mean(), np.median(ae), ae, fitted

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
stresses = sorted(raw.keys())

# ── MCEM run ──────────────────────────────────────────────────────
# Init from SEV K=2 NPMLE estimates (faster convergence than random)
b0, b1, sig   = -8.375, -6.389, 0.425
mu_d, sig_d   = np.log(0.59), 0.05     # near NPMLE Delta atoms
M_samples     = 200   # MC samples per observation per iteration
max_iter      = 80
tol           = 5e-4
master_seed   = 2025

print("="*65)
print(f"MCEM: SEV + LogNormal g(Delta)  M={M_samples} samples/obs/iter")
print("="*65)
print(f"Init: b0={b0:.4f}  b1={b1:.4f}  sig={sig:.4f}  mu_d={mu_d:.4f}  sig_d={sig_d:.4f}")
print()

ll_prev = -np.inf
all_samples = None

for it in range(max_iter):
    rng = np.random.default_rng(master_seed + it * 1000)

    # E-step
    all_samples = [
        sample_delta_posterior(Y_r[i], S_r[i], b0, b1, sig, mu_d, sig_d, M_samples, rng)
        for i in range(n)
    ]
    n_rej = sum(1 for s in all_samples if len(s) < M_samples // 2)

    # M-step
    b0, b1, sig, mu_d, sig_d = m_step(Y_r, S_r, cens_r, all_samples, b0, b1, sig)

    # log-lik
    ll = mc_loglik(Y_r, S_r, cens_r, b0, b1, sig, mu_d, sig_d, all_samples)
    E_delta = np.exp(mu_d + sig_d**2 / 2)

    print(f"  iter={it+1:2d}  b0={b0:8.4f}  b1={b1:8.4f}  sig={sig:.4f}"
          f"  mu_d={mu_d:.4f}  sig_d={sig_d:.4f}  E[D]={E_delta:.4f}"
          f"  ll={ll:.4f}  grid_fb={n_rej}")

    if it > 0 and abs(ll - ll_prev) < tol:
        print(f"  Converged at iter {it+1}")
        break
    ll_prev = ll

# ── final ASSE ────────────────────────────────────────────────────
print()
print("="*65)
print("FINAL RESULTS")
print("="*65)
print(f"  b0={b0:.5f}  b1={b1:.5f}  sig={sig:.5f}")
print(f"  mu_d={mu_d:.5f}  sig_d={sig_d:.5f}  E[Delta]={np.exp(mu_d+sig_d**2/2):.5f}")
print(f"  MC log-lik = {ll:.4f}")
print()

asse, mae, medae, ae, fitted = asse_from_samples(Y_r, S_r, b0, b1, sig, all_samples)
print(f"  ASSE  = {asse:.4f}  (target: INLA 8.63, Chiu thesis 10.80)")
print(f"  MAE   = {mae:.5f}")
print(f"  MedAE = {medae:.5f}")

print()
print(f"Per-stress ASSE:")
print(f"  {'S':>6} {'n':>4} {'ASSE':>8} {'MAE':>8} {'INLA_MAE':>10}")
inla_mae = {0.675:0.0712, 0.75:0.0884, 0.825:0.1165, 0.9:0.1268, 0.95:0.1720}
for s in stresses:
    idx = S_r == s
    print(f"  {s:>6.3f} {idx.sum():>4d} {ae[idx].sum():>8.4f} {ae[idx].mean():>8.4f} {inla_mae[s]:>10.4f}")

print()
print("Reference ASSE table:")
print(f"  INLA (Normal+GH)         : {8.63:.2f}")
print(f"  Chiu (2005) thesis       : {10.80:.2f}")
print(f"  Pascual & Meeker (1999)  : {12.84:.2f}")
print(f"  Normal NPMLE K=4         : {11.71:.2f}")
print(f"  SEV NPMLE K=4            : {13.64:.2f}")
print(f"  SEV MCEM (this)          : {asse:.2f}")
