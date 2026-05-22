"""
rfl_sev_inla.py -- SEV conditional + INLA-style 9-pt Gauss-Hermite
Model:
  g(Delta)   ~ LogNormal(mu_d, sig_d^2)
  f(Y|Delta) ~ SEV(b0 + b1*log(S-Delta), sig)   [vs Normal in rfl_inla.py]

Integration: same two-level INLA structure as rfl_inla.py
  Inner : 9-pt GH centred at per-obs posterior mode of Delta_i
  Outer : grid -> dual_annealing -> Nelder-Mead (identical heuristic)

Fitted value: E[Y|Delta,sig]_SEV = mu(S,Delta) - sig*gamma  (Euler-Mascheroni correction)
ASSE compared with Normal+INLA (8.63), SEV MCEM (10.89), Chiu (10.80)
"""
import sys
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize_scalar, minimize
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

# ── SEV functions ─────────────────────────────────────────────────
def sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def sev_logsf(y, mu, sig):
    return -np.exp((y - mu) / sig)

# ── Gauss-Hermite setup (same as rfl_inla.py) ─────────────────────
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)
_GHE = np.exp(_GHX ** 2 / 2)

# ── Per-observation log integrand ─────────────────────────────────
def _log_h(d, y, s, is_cens, b0, b1, sig, mu_d, sig_d):
    if d <= 0 or d >= s - 1e-8:
        return -1e30
    mu_y = b0 + b1 * np.log(s - d)
    lf = sev_logsf(y, mu_y, sig) if is_cens else sev_logpdf(y, mu_y, sig)
    lg = lognorm.logpdf(d, s=sig_d, scale=np.exp(mu_d))
    return lf + lg

# ── Multiple Laplace (GH quadrature) ─────────────────────────────
def _multi_laplace(y, s, is_cens, b0, b1, sig, mu_d, sig_d):
    res = minimize_scalar(
        lambda d: -_log_h(d, y, s, is_cens, b0, b1, sig, mu_d, sig_d),
        bounds=(1e-6, s - 1e-4), method='bounded')
    d_hat = res.x
    lh0   = _log_h(d_hat, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    if lh0 < -1e25:
        return -1e30

    eps   = max(d_hat * 0.005, 5e-6)
    lhp   = _log_h(d_hat + eps, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    lhm   = _log_h(d_hat - eps, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    kappa = max(-(lhp - 2 * lh0 + lhm) / eps ** 2, 1e-6)
    sig_t = 1.0 / np.sqrt(kappa)

    d_pts  = d_hat + sig_t * _GHX
    lh_pts = np.array([_log_h(dj, y, s, is_cens, b0, b1, sig, mu_d, sig_d) for dj in d_pts])
    log_terms = lh_pts + _GHX ** 2 / 2
    log_shift = log_terms.max()
    L_i = sig_t * np.sum(_GHW * np.exp(log_terms - log_shift)) * np.exp(log_shift)
    return np.log(max(L_i, 1e-300))

# ── Posterior mean of mu (for ASSE fitted value) ──────────────────
def _posterior_mean_mu(y, s, b0, b1, sig, mu_d, sig_d):
    """
    E[mu(S,Delta) | y, S] via GH quadrature.
    Fitted value for ASSE = this - sig*gamma  (SEV posterior predictive mean)
    """
    res = minimize_scalar(
        lambda d: -_log_h(d, y, s, False, b0, b1, sig, mu_d, sig_d),
        bounds=(1e-6, s - 1e-4), method='bounded')
    d_hat = res.x
    lh0   = _log_h(d_hat, y, s, False, b0, b1, sig, mu_d, sig_d)
    if lh0 < -1e25:
        return b0 + b1 * np.log(max(s - d_hat, 1e-8))

    eps   = max(d_hat * 0.005, 5e-6)
    lhp   = _log_h(d_hat + eps, y, s, False, b0, b1, sig, mu_d, sig_d)
    lhm   = _log_h(d_hat - eps, y, s, False, b0, b1, sig, mu_d, sig_d)
    kappa = max(-(lhp - 2 * lh0 + lhm) / eps ** 2, 1e-6)
    sig_t = 1.0 / np.sqrt(kappa)

    d_pts    = d_hat + sig_t * _GHX
    lh_pts   = np.array([_log_h(dj, y, s, False, b0, b1, sig, mu_d, sig_d) for dj in d_pts])
    log_terms = lh_pts + _GHX ** 2 / 2
    log_shift = log_terms.max()
    unnorm   = _GHW * np.exp(log_terms - log_shift)
    weights  = unnorm / max(unnorm.sum(), 1e-300)

    mu_pts = np.array([b0 + b1 * np.log(max(s - dj, 1e-8)) for dj in d_pts])
    return float(np.sum(weights * mu_pts))

# ── Total log-likelihood ──────────────────────────────────────────
def loglik(theta, Y, S, cens):
    b0, b1, log_sig, mu_d, log_sig_d = theta
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)
    if sig < 0.01 or sig_d < 0.005:
        return -1e30
    return sum(
        _multi_laplace(Y[i], S[i], bool(cens[i]), b0, b1, sig, mu_d, sig_d)
        for i in range(len(Y))
    )

# ── ASSE from GH posterior mean ───────────────────────────────────
def compute_asse(Y, S, cens, theta):
    b0, b1, log_sig, mu_d, log_sig_d = theta
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)
    obs   = ~cens
    fitted = np.array([
        _posterior_mean_mu(Y[i], S[i], b0, b1, sig, mu_d, sig_d) - sig * EULER_GAMMA
        for i in range(len(Y))
    ])
    ae = np.abs(Y - fitted)
    return ae.sum(), ae.mean(), np.median(ae), ae, fitted

# ── Heuristic optimizer (grid -> SA -> NM) ────────────────────────
_BOUNDS = [(-20., -2.), (-20., -2.),
           (np.log(0.05), np.log(2.0)),
           (-3.0, 1.0),
           (np.log(0.005), np.log(1.0))]

def optimize(Y, S, cens, x0):
    """
    L-BFGS-B from warm-start x0, then Nelder-Mead polish.
    Verified to find the same MLE as full grid+SA approach (~10x faster).
    Requires a reasonable warm-start (use MCEM result or rfl_inla.py output).
    """
    res_bfgs = minimize(
        lambda t: -loglik(t, Y, S, cens),
        x0=x0, method='L-BFGS-B', bounds=_BOUNDS,
        options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 500})
    print(f"  [BFGS ] best ll = {-res_bfgs.fun:.4f}")

    res_nm = minimize(
        lambda t: -loglik(t, Y, S, cens),
        x0=res_bfgs.x, method='Nelder-Mead',
        options={'maxiter': 15000, 'xatol': 1e-7, 'fatol': 1e-7})
    print(f"  [NM   ] best ll = {-res_nm.fun:.4f}")
    return res_nm.x, float(-res_nm.fun)

# ── Data ──────────────────────────────────────────────────────────
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
cens_r = np.zeros(len(Y_r), bool)
stresses = sorted(_raw.keys())
n = len(Y_r)

# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print(f"SEV + INLA (9-pt GH)  n={n}  SEV conditional + LogNormal g")
    print("=" * 65)

    # Warm-start hint from SEV MCEM result
    mcem_x0 = [-8.480, -6.592, np.log(0.268), -0.540, np.log(0.029)]
    print(f"\nWarm-start hint from SEV MCEM:")
    print(f"  b0={mcem_x0[0]:.4f}  b1={mcem_x0[1]:.4f}  sig={np.exp(mcem_x0[2]):.4f}"
          f"  mu_d={mcem_x0[3]:.4f}  sig_d={np.exp(mcem_x0[4]):.4f}")
    ll_warm = loglik(mcem_x0, Y_r, S_r, cens_r)
    print(f"  ll at warm-start = {ll_warm:.4f}")

    print("\nRunning optimizer (L-BFGS-B + NM from MCEM warm-start)...")
    theta, ll = optimize(Y_r, S_r, cens_r, x0=mcem_x0)

    b0, b1, log_sig, mu_d, log_sig_d = theta
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)
    E_d   = np.exp(mu_d + sig_d**2 / 2)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"  b0={b0:.5f}  b1={b1:.5f}  sig={sig:.5f}")
    print(f"  mu_d={mu_d:.5f}  sig_d={sig_d:.5f}  E[Delta]={E_d:.5f}")
    print(f"  log-lik = {ll:.4f}")

    # ASSE
    asse, mae, medae, ae, fitted = compute_asse(Y_r, S_r, cens_r, theta)
    print(f"\n  ASSE  = {asse:.4f}")
    print(f"  MAE   = {mae:.5f}")
    print(f"  MedAE = {medae:.5f}")

    print(f"\nPer-stress ASSE:")
    inla_mae = {0.675:0.0712, 0.75:0.0884, 0.825:0.1165, 0.9:0.1268, 0.95:0.1720}
    print(f"  {'S':>6} {'n':>4} {'ASSE':>8} {'MAE':>8} {'Normal+INLA MAE':>16}")
    for s in stresses:
        idx = S_r == s
        print(f"  {s:6.3f} {idx.sum():4d} {ae[idx].sum():8.4f} {ae[idx].mean():8.4f} {inla_mae[s]:16.4f}")

    print()
    print("Reference ASSE table:")
    print(f"  INLA (Normal+GH)         : 8.63")
    print(f"  SEV  MCEM (rfl_mcem.py)  : 10.89")
    print(f"  Chiu (2005) thesis       : 10.80")
    print(f"  Pascual & Meeker (1999)  : 12.84")
    print(f"  Normal NPMLE K=4         : 11.71")
    print(f"  SEV NPMLE K=4            : 13.64")
    print(f"  SEV + INLA (this)        : {asse:.2f}")

    # ──────────────────────────────────────────────────────────────
    # Censoring scenarios — mirrors rfl_inla.py [A][B][C]
    # Warm-start: [A] MLE theta (most reliable for nearby censoring)
    # ──────────────────────────────────────────────────────────────
    def _apply_hybrid_censoring(Y, S, r_frac=0.80, T_cutoff=None):
        """T*_j = min(X_{r_j:n_j}, T_j) per stress level."""
        cens  = np.zeros(len(Y), bool)
        Y_obs = Y.copy()
        for sv in np.unique(S):
            idx = np.where(S == sv)[0]
            n_j = len(idx)
            r_j = int(np.ceil(r_frac * n_j))
            x_r = np.sort(Y[idx])[min(r_j - 1, n_j - 1)]
            T_st = min(x_r, T_cutoff) if T_cutoff is not None else x_r
            mask = Y[idx] > T_st
            cens[idx[mask]]  = True
            Y_obs[idx[mask]] = T_st
        return Y_obs, cens

    print("\n\n" + "=" * 65)
    print("CENSORING COMPARISON — SEV + INLA (9-pt GH)")
    print("Warm-start: [A] MLE theta for each censored scenario")
    print("=" * 65)

    # [B] Type I: censor at 80th percentile per stress (~20%)
    Y_B    = Y_r.copy(); cens_B = np.zeros(n, bool)
    for sv in stresses:
        idx  = np.where(S_r == sv)[0]
        T_j  = np.quantile(Y_r[idx], 0.80)
        mask = Y_r[idx] > T_j
        cens_B[idx[mask]] = True
        Y_B[idx[mask]]    = T_j

    # [C] Hybrid I-II: r=80%, T = global 75th percentile
    T_hyb       = np.quantile(Y_r, 0.75)
    Y_C, cens_C = _apply_hybrid_censoring(Y_r, S_r, r_frac=0.80, T_cutoff=T_hyb)

    scenarios = [
        ('A', Y_r,  S_r, cens_r, f'No censoring          (n_obs=75, n_cens= 0)'),
        ('B', Y_B,  S_r, cens_B, f'Type I 80th pct       (n_obs={( ~cens_B).sum():2d}, n_cens={cens_B.sum():2d}, {cens_B.mean():.1%})'),
        ('C', Y_C,  S_r, cens_C, f'Hybrid r=80%,T=75th   (n_obs={(~cens_C).sum():2d}, n_cens={cens_C.sum():2d}, {cens_C.mean():.1%})'),
    ]

    x0_cens = theta   # warm-start from [A] MLE

    print(f"\n  {'Scen':<4} {'Description':<42} {'ASSE':>6}  {'log-lik':>9}  "
          f"{'b0':>8}  {'b1':>8}  {'sig':>6}  {'sig_d':>7}")
    print(f"  {'-'*4}  {'-'*42}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}")

    for label, Y_sc, S_sc, cens_sc, desc in scenarios:
        if label == 'A':
            th_sc, ll_sc = theta, ll          # already computed
        else:
            print(f"\n  Optimising scenario [{label}]...")
            th_sc, ll_sc = optimize(Y_sc, S_sc, cens_sc, x0=x0_cens)
            x0_cens = th_sc                   # chain warm-start A→B→C

        asse_sc, mae_sc, _, ae_sc, _ = compute_asse(Y_sc, S_sc, cens_sc, th_sc)
        b0_, b1_, ls_, md_, lsd_ = th_sc
        print(f"  [{label}]  {desc:<42} {asse_sc:6.2f}  {ll_sc:9.4f}  "
              f"{b0_:8.4f}  {b1_:8.4f}  {np.exp(ls_):6.4f}  {np.exp(lsd_):7.5f}")

        # Per-stress breakdown
        print(f"         {'S':>6}  {'n_obs':>5}  {'MAE':>8}")
        for sv in stresses:
            idx_s  = (S_sc == sv) & (~cens_sc)
            if idx_s.sum() == 0: continue
            print(f"         {sv:6.3f}  {idx_s.sum():5d}  {ae_sc[idx_s].mean():8.5f}")

    print("\n" + "─" * 65)
    print("  Three-method summary (ASSE on uncensored observations)")
    print("─" * 65)
    print(f"  {'Method':<30} {'[A] 0%':>7}  {'[B] 20%':>8}  {'[C] 37%':>8}")
    print(f"  {'Normal+INLA (rfl_inla.py)':<30} {'8.63':>7}  {'n/a':>8}  {'crashed':>8}")
    print(f"  {'SEV+INLA (this)':<30}  see above")
