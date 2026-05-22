"""
rfl_burr_em.py — Burr XII 封閉形 × EM-GMM prior on Δ_i
=============================================================
Model:
  Δ_i ~ Σ_k π_k N(μ_k, σ_k²)             [K-component GMM prior, EM 更新]
  U_i | Δ_i ~ Gamma(a, a/(S_i-Δ_i)^α)    [Gamma conjugate, analytical]
  Y_i | U_i ~ SEV(b0 - σ·log(U_i), σ)    [conditional SEV]

Marginalizing U_i analytically → Burr XII closed-form:
  f_Burr(y_i | Δ_i=δ) = (a^{a+1}/σ) · exp(w) / (a + exp(w))^{a+1}
  w(δ) = (y_i - b0 - b1·log(S_i - δ)) / σ

EM Algorithm:
  E-step: trapezoidal grid on Δ ∈ (0, S_min=0.675)
    log L_{ik} = log ∫ f_Burr(y_i|δ) N(δ;μ_k,σ_k²) dδ
    r_{ik} ∝ π_k · L_{ik}                  [soft assignments]
    E[Δ|Y,Z=k], E[Δ²|Y,Z=k]               [component posterior moments]

  M-step:
    π_k, μ_k, σ_k : closed-form GMM update (standard EM formula)
    b0, b1, σ, a  : L-BFGS-B on mixture marginal log-lik

Fitted value:
  E[Y^new | Y_i] = Σ_k r_{ik} · ∫ E[Y^new|Δ_i=δ] p(Δ_i|Y_i,Z_i=k) dδ
  E[Y^new|Δ_i=δ] = μ*(δ) - σ·(ψ(a+1) - log(a+exp(w(δ))) + γ_E)

Parameters: 4 likelihood (b0,b1,σ,a) + (3K-1) GMM = 3+3K total
  K=1: 6 params  (comparable to Burr+INLA w/ Gaussian prior)
  K=2: 9 params  (more flexible prior)
  K=3: 12 params
"""
import numpy as np
from scipy.special import digamma
from scipy.optimize import minimize
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

# ── Data (n=75, 5 stress levels) ─────────────────────────────
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
    for y in ys:
        S_r.append(s); Y_r.append(np.log(y))
S_r = np.array(S_r); Y_r = np.array(Y_r)
stresses = sorted(_raw.keys())
n = len(Y_r)

# ── Global integration grid: Δ ∈ (0.002, 0.670) ─────────────
# Bounded by min(S)=0.675 with margin; 400 uniform points
N_GRID  = 400
_DGRID  = np.linspace(0.002, 0.670, N_GRID)
_DW     = _DGRID[1] - _DGRID[0]      # uniform spacing for trapezoidal rule
_DGRID2 = _DGRID ** 2                 # precomputed for M2 moments

# ── Burr XII conditional log-pdf (vectorized over grid) ──────
def _burr_logpdf_vec(y, s, b0, b1, sig, log_a):
    """
    log f_Burr(y | Δ=δ) for all δ in _DGRID.
    Returns array shape (N_GRID,); invalid entries (δ >= s) = -1e30.
    """
    a   = np.exp(log_a)
    out = np.full(N_GRID, -1e30)
    valid = _DGRID < s - 1e-8
    if not valid.any():
        return out
    d       = _DGRID[valid]
    mu_star = b0 + b1 * np.log(s - d)
    w       = (y - mu_star) / sig
    log_apt = np.logaddexp(log_a, w)          # log(a + exp(w)), stable
    out[valid] = (a+1)*log_a - np.log(sig) + w - (a+1)*log_apt
    return out

# ── Burr XII conditional log-survival (vectorized over grid) ─
def _burr_logsf_vec(y, s, b0, b1, sig, log_a):
    """
    log S_Burr(y | Δ=δ) for all δ in _DGRID.  (Right-censoring contribution)
    Derivation: S_Burr(y|Δ) = (a / (a + exp(w)))^a
      => log S = a * log_a - a * logaddexp(log_a, w)
    Returns array shape (N_GRID,); invalid entries = -1e30.
    """
    a   = np.exp(log_a)
    out = np.full(N_GRID, -1e30)
    valid = _DGRID < s - 1e-8
    if not valid.any():
        return out
    d       = _DGRID[valid]
    mu_star = b0 + b1 * np.log(s - d)
    w       = (y - mu_star) / sig
    log_apt = np.logaddexp(log_a, w)
    out[valid] = a * (log_a - log_apt)
    return out

# ── E[Y^new | Δ=δ] on grid ───────────────────────────────────
def _yhat_vec(y, s, b0, b1, sig, log_a):
    """
    E[Y^new | Δ_i=δ] for all δ in _DGRID (posterior predictive given Δ).
    Fallback value used for invalid entries.
    """
    a        = np.exp(log_a)
    fallback = b0 + b1 * np.log(max(s - 0.53, 1e-8)) - sig * EULER_GAMMA
    out      = np.full(N_GRID, fallback)
    valid    = _DGRID < s - 1e-8
    d        = _DGRID[valid]
    mu_star  = b0 + b1 * np.log(s - d)
    w        = (y - mu_star) / sig
    log_apt  = np.logaddexp(log_a, w)
    out[valid] = mu_star - sig * (digamma(a+1) - log_apt + EULER_GAMMA)
    return out

# ── GMM log-density on grid: shape (K, N_GRID) ───────────────
def _gmm_logphi(mus, sigs, K):
    """log N(δ; μ_k, σ_k²) for each component k. Returns (K, N_GRID)."""
    log_phi = np.zeros((K, N_GRID))
    for k in range(K):
        log_phi[k] = (
            -0.5 * ((_DGRID - mus[k]) / sigs[k])**2
            - np.log(sigs[k]) - 0.5 * np.log(2 * np.pi)
        )
    return log_phi

# ── E-step ───────────────────────────────────────────────────
def e_step(Y, S, b0, b1, sig, log_a, pis, mus, sigs, K, cens=None):
    """
    Full E-step over all n observations.

    cens : bool array (n,) or None — True = right-censored observation.
           Censored obs use log S_Burr instead of log f_Burr.

    Returns:
      R        : (n, K) soft component assignments
      MU_P     : (n, K) E[Δ_i | Y_i, Z_i=k]   (posterior mean per component)
      M2_P     : (n, K) E[Δ_i² | Y_i, Z_i=k]  (posterior 2nd moment)
      YHAT_K   : (n, K) E[Y_i^new | Y_i, Z_i=k]  (only meaningful for uncensored)
      total_ll : float, log p(Y) = Σ_i log Σ_k π_k L_{ik}
    """
    log_phi  = _gmm_logphi(mus, sigs, K)           # (K, N_GRID)
    log_pis  = np.log(np.clip(pis, 1e-300, None))  # (K,)

    n_obs    = len(Y)
    R        = np.zeros((n_obs, K))
    MU_P     = np.zeros((n_obs, K))
    M2_P     = np.zeros((n_obs, K))
    YHAT_K   = np.zeros((n_obs, K))
    total_ll = 0.0

    for i in range(n_obs):
        is_cens = bool(cens[i]) if cens is not None else False
        if is_cens:
            log_f = _burr_logsf_vec(Y[i], S[i], b0, b1, sig, log_a)  # (N_GRID,)
        else:
            log_f = _burr_logpdf_vec(Y[i], S[i], b0, b1, sig, log_a) # (N_GRID,)
        yhat_g = _yhat_vec(Y[i], S[i], b0, b1, sig, log_a)            # (N_GRID,)
        valid  = _DGRID < S[i] - 1e-8                               # (N_GRID,)

        log_lik_k = np.full(K, -1e30)

        for k in range(K):
            lh = np.where(valid, log_f + log_phi[k], -1e30)  # (N_GRID,)
            shift = lh.max()
            if shift < -1e25:
                continue
            eh = np.exp(lh - shift)
            eh[~valid] = 0.0
            Zk = _DW * eh.sum()
            if Zk < 1e-300:
                continue
            log_lik_k[k] = shift + np.log(Zk)

            # Posterior weights on grid (uniform spacing → wts ∝ eh)
            wts = eh / eh.sum()
            MU_P[i, k]   = wts @ _DGRID
            M2_P[i, k]   = wts @ _DGRID2
            YHAT_K[i, k] = wts @ yhat_g

        # Soft assignments: r_{ik} ∝ π_k · L_{ik}
        log_r = log_pis + log_lik_k
        obs_ll = np.logaddexp.reduce(log_r)
        total_ll += obs_ll
        r = np.exp(log_r - log_r.max())
        r /= r.sum()
        R[i] = r

    return R, MU_P, M2_P, YHAT_K, total_ll

# ── M-step: GMM closed-form update ───────────────────────────
def m_step_gmm(R, MU_P, M2_P, K, min_sig=0.015):
    """
    Standard EM update for GMM parameters.
    μ_k^new  = Σ_i r_{ik} E[Δ|Y,Z=k] / Σ_i r_{ik}
    σ_k^2new = Σ_i r_{ik} E[Δ²|Y,Z=k] / Σ_i r_{ik}  - μ_k^2
    π_k^new  = (Σ_i r_{ik}) / n
    """
    Rk   = R.sum(axis=0).clip(1e-6)
    pis  = Rk / Rk.sum()
    mus  = (R * MU_P).sum(axis=0) / Rk
    m2   = (R * M2_P).sum(axis=0) / Rk
    sigs = np.sqrt(np.maximum(m2 - mus**2, min_sig**2))
    sigs = np.maximum(sigs, min_sig)
    return pis, mus, sigs

# ── M-step: likelihood params (b0, b1, σ, a) ─────────────────
def _neg_mll(theta4, Y, S, pis, mus, sigs, K, log_phi, cens=None):
    """Negative mixture marginal log-likelihood for scipy.optimize."""
    b0, b1, log_sig, log_a = theta4
    sig = np.exp(log_sig)
    a   = np.exp(log_a)
    if sig < 0.01 or a < 0.05:
        return 1e30
    log_pis = np.log(np.clip(pis, 1e-300, None))
    total   = 0.0

    for i in range(len(Y)):
        is_cens = bool(cens[i]) if cens is not None else False
        if is_cens:
            log_f = _burr_logsf_vec(Y[i], S[i], b0, b1, sig, log_a)
        else:
            log_f = _burr_logpdf_vec(Y[i], S[i], b0, b1, sig, log_a)
        valid = _DGRID < S[i] - 1e-8
        log_r = np.full(K, -1e30)
        for k in range(K):
            lh    = np.where(valid, log_f + log_phi[k], -1e30)
            shift = lh.max()
            if shift < -1e25:
                continue
            eh = np.exp(lh - shift)
            eh[~valid] = 0.0
            Zk = _DW * eh.sum()
            if Zk > 1e-300:
                log_r[k] = log_pis[k] + shift + np.log(Zk)
        total += np.logaddexp.reduce(log_r)

    return -total

def m_step_lik(b0, b1, log_sig, log_a, Y, S, pis, mus, sigs, K,
               fix_sig=False, fix_a=False, cens=None):
    """
    Optimize likelihood params (b0, b1, sig, a) given fixed GMM.
    fix_sig : if True, keep sig fixed (prevents sig->0 degenerate solution)
    fix_a   : if True, keep a fixed

    Note: the degenerate solution sig->0, a->inf exists whenever the
    trapezoidal grid can 'find' each observation's implied delta_i*.
    Fixing sig (e.g., at SEV+INLA value 0.19) prevents this collapse.
    """
    log_phi = _gmm_logphi(mus, sigs, K)

    if fix_sig and fix_a:
        # Only optimize (b0, b1)
        def obj(t2):
            b0_, b1_ = t2
            theta4 = np.array([b0_, b1_, log_sig, log_a])
            return _neg_mll(theta4, Y, S, pis, mus, sigs, K, log_phi, cens)
        res = minimize(obj, x0=[b0, b1], method='L-BFGS-B',
                       bounds=[(-20., -2.), (-20., -2.)],
                       options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 300})
        return np.array([res.x[0], res.x[1], log_sig, log_a]), -res.fun

    elif fix_sig:
        # Optimize (b0, b1, log_a) with sig fixed
        def obj(t3):
            b0_, b1_, la_ = t3
            return _neg_mll([b0_, b1_, log_sig, la_], Y, S, pis, mus, sigs, K, log_phi, cens)
        bounds3 = [(-20.,-2.), (-20.,-2.), (np.log(0.05), np.log(50.))]
        res = minimize(obj, x0=[b0, b1, log_a], method='L-BFGS-B', bounds=bounds3,
                       options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 300})
        return np.array([res.x[0], res.x[1], log_sig, res.x[2]]), -res.fun

    else:
        # Full 4-parameter optimization
        # Tighter bounds to prevent degenerate sig->0, a->inf solution:
        #   sig >= 0.15 (motivated by SEV+INLA result: sig=0.190)
        #   a   <= 50   (prevents pure SEV limit collapsing the Gamma mixing)
        x0 = np.array([b0, b1, log_sig, log_a])
        bounds = [
            (-20., -2.),                           # b0
            (-20., -2.),                           # b1
            (np.log(0.15), np.log(2.0)),           # log_sig  (>= 0.15)
            (np.log(1.00), np.log(50.)),           # log_a    (>= 1 prevents S_Burr->1 under censoring)
        ]
        res = minimize(
            lambda t: _neg_mll(t, Y, S, pis, mus, sigs, K, log_phi, cens),
            x0=x0, method='L-BFGS-B', bounds=bounds,
            options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 300}
        )
        return res.x, -res.fun

# ── Full EM fit ───────────────────────────────────────────────
def em_fit(Y, S, K=2, n_iter=60, tol=5e-5, verbose=True,
           fix_sig=False, fix_a=False, cens=None):
    """
    EM-GMM fitting for Burr XII + GMM prior.

    Y, S  : log-life and stress arrays (n,)
    cens  : bool array (n,) or None — True = right-censored.
            Censored obs use log S_Burr in likelihood; excluded from ASSE.

    Init: SEV+INLA result for (b0,b1,sig,a); BIC K=2 means for mu_k.

    fix_sig: True => keep sig fixed at init value (prevents sig->0 collapse)
    fix_a  : True => keep a fixed at init value

    Modes:
      fix_sig=False, fix_a=False  => full 4-param M-step (bounds tightened)
      fix_sig=True,  fix_a=False  => optimize (b0,b1,a); sig fixed
      fix_sig=True,  fix_a=True   => optimize (b0,b1) only; sig,a fixed

    Returns dict with fitted params, ASSE (uncensored only), AIC, BIC.
    """
    # Initialize likelihood params from SEV+INLA / Burr+INLA
    b0, b1   = -9.316, -8.534
    log_sig  = np.log(0.190)   # SEV+INLA result
    log_a    = np.log(23.2)    # Burr+INLA result
    sig      = np.exp(log_sig)

    # Initialize GMM from BIC K=2 profile-lik result: Delta-hat ≈ (0.532, 0.569)
    if K == 1:
        mus  = np.array([0.532])
        sigs = np.array([0.06])
        pis  = np.array([1.0])
    elif K == 2:
        mus  = np.array([0.50, 0.58])
        sigs = np.array([0.05, 0.05])
        pis  = np.array([0.50, 0.50])
    else:
        mus  = np.linspace(0.43, 0.63, K)
        sigs = np.full(K, 0.05)
        pis  = np.full(K, 1.0 / K)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"EM-GMM K={K}  init: b0={b0:.3f} b1={b1:.3f} "
              f"sig={sig:.3f} a={np.exp(log_a):.1f}")
        print(f"  mu_k={np.round(mus,4)}  sig_k={np.round(sigs,4)}  pi_k={np.round(pis,3)}")
        print(f"{'─'*60}")

    prev_ll = -np.inf
    for it in range(n_iter):
        # ── E-step ──────────────────────────────────────────
        R, MU_P, M2_P, YHAT_K, ll = e_step(Y, S, b0, b1, sig, log_a, pis, mus, sigs, K, cens)
        delta_ll = ll - prev_ll

        if verbose:
            print(f"  iter {it+1:3d} | ll={ll:10.4f}  Δll={delta_ll:+.6f} | "
                  f"mu={np.round(mus,4)}  pi={np.round(pis,3)}"
                  f"  sig={sig:.4f}  a={np.exp(log_a):.3f}")

        if it >= 3 and abs(delta_ll) < tol:
            if verbose:
                print(f"  [Converged at iter {it+1}]")
            break
        prev_ll = ll

        # ── M-step: GMM params (closed form) ────────────────
        pis, mus, sigs = m_step_gmm(R, MU_P, M2_P, K)

        # ── M-step: likelihood params (numerical) ───────────
        theta4, mll = m_step_lik(b0, b1, log_sig, log_a, Y, S, pis, mus, sigs, K,
                                  fix_sig=fix_sig, fix_a=fix_a, cens=cens)
        b0, b1, log_sig, log_a = theta4
        sig = np.exp(log_sig)

    # ── Final E-step for ASSE computation ───────────────────
    R, MU_P, M2_P, YHAT_K, ll = e_step(Y, S, b0, b1, sig, log_a, pis, mus, sigs, K, cens)
    yhat = (R * YHAT_K).sum(axis=1)
    ae   = np.abs(Y - yhat)
    obs  = ~cens if cens is not None else np.ones(len(Y), bool)
    asse = ae[obs].sum()

    # ── Information criteria ─────────────────────────────────
    # Free params: b0,b1,sig,a (4) + (K-1) free pi + K mu + K sig = 3+3K
    n_params = 3 + 3 * K
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(len(Y))

    return dict(
        b0=b0, b1=b1, sig=sig, a=np.exp(log_a), log_a=log_a,
        pis=pis, mus=mus, sigs=sigs,
        loglik=ll, yhat=yhat, ae=ae, asse=asse,
        n_params=n_params, aic=aic, bic=bic, R=R
    )

# ── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print(f"rfl_burr_em.py  n={n}  Burr XII x EM-GMM prior on Delta_i")
    print("  f_Burr(y|Delta_i) closed-form; GMM prior updated by EM")
    print("=" * 65)

    sev_mae = {0.675: 0.0483, 0.75: 0.0583, 0.825: 0.0755,
               0.9: 0.0765, 0.95: 0.1252}

    # ── Mode A: free (sig, a) with tighter bounds ──────────────
    print("\n" + "=" * 65)
    print("Mode A: free sig (>=0.15) and a (<=50) — tighter bounds")
    print("=" * 65)
    res_a = {}
    for K in [1, 2]:
        print(f"\n--- K={K} ---")
        res_a[K] = em_fit(Y_r, S_r, K=K, n_iter=60, verbose=True,
                          fix_sig=False, fix_a=False)

    # ── Mode B: fix sig at SEV+INLA value (0.190) ─────────────
    print("\n" + "=" * 65)
    print("Mode B: sig fixed at 0.190 (SEV+INLA); optimize (b0,b1,a)")
    print("=" * 65)
    res_b = {}
    for K in [1, 2, 3]:
        print(f"\n--- K={K} ---")
        res_b[K] = em_fit(Y_r, S_r, K=K, n_iter=60, verbose=True,
                          fix_sig=True, fix_a=False)

    # ── Mode C: fix (sig, a) — only GMM prior updated ──────────
    print("\n" + "=" * 65)
    print("Mode C: sig=0.190, a=23.2 fixed; only (b0,b1) + GMM updated")
    print("=" * 65)
    res_c = {}
    for K in [1, 2, 3]:
        print(f"\n--- K={K} ---")
        res_c[K] = em_fit(Y_r, S_r, K=K, n_iter=60, verbose=True,
                          fix_sig=True, fix_a=True)

    # ── Comparison table ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("ASSE / AIC Comparison (all modes)")
    print("=" * 65)
    print(f"  {'Model':<36} {'ASSE':>6}  {'log-lik':>9}  {'AIC':>8}  {'n_p':>4}")
    print(f"  {'-'*36}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*4}")
    print(f"  {'SEV + INLA (best)':<36} {'5.76':>6}  {'-72.88':>9}  {'155.76':>8}  {'5':>4}")
    print(f"  {'Burr XII + INLA':<36} {'5.74':>6}  {'-72.88':>9}  {'157.77':>8}  {'6':>4}")
    for K in [1, 2]:
        r = res_a[K]
        label = f"Mode A: free sig/a  K={K}"
        print(f"  {label:<36} {r['asse']:6.2f}  {r['loglik']:9.4f}  "
              f"{r['aic']:8.2f}  {r['n_params']:>4}"
              f"  sig={r['sig']:.3f} a={r['a']:.1f}")
    for K in [1, 2, 3]:
        r = res_b[K]
        label = f"Mode B: fix_sig     K={K}"
        print(f"  {label:<36} {r['asse']:6.2f}  {r['loglik']:9.4f}  "
              f"{r['aic']:8.2f}  {r['n_params']:>4}"
              f"  sig={r['sig']:.3f} a={r['a']:.1f}")
    for K in [1, 2, 3]:
        r = res_c[K]
        label = f"Mode C: fix_sig+a   K={K}"
        print(f"  {label:<36} {r['asse']:6.2f}  {r['loglik']:9.4f}  "
              f"{r['aic']:8.2f}  {r['n_params']:>4}"
              f"  sig={r['sig']:.3f} a={r['a']:.1f}")

    # ── Per-stress for Mode C K=2 (most comparable to Burr+INLA) ──
    print("\n" + "=" * 65)
    print("Per-stress detail: Mode C K=2 vs SEV+INLA")
    print("=" * 65)
    r = res_c[2]
    ae = r['ae']
    print(f"  b0={r['b0']:.5f}  b1={r['b1']:.5f}  sig={r['sig']:.5f}  a={r['a']:.3f}")
    for k in range(2):
        print(f"  Comp {k+1}: mu={r['mus'][k]:.5f}  sig_k={r['sigs'][k]:.5f}  pi={r['pis'][k]:.4f}")
    print(f"  {'S':>6}  {'n':>4}  {'ASSE':>8}  {'MAE':>8}  {'SEV+INLA MAE':>14}")
    for s_val in stresses:
        idx = S_r == s_val
        print(f"  {s_val:6.3f}  {idx.sum():4d}  {ae[idx].sum():8.4f}  "
              f"{ae[idx].mean():8.5f}  {sev_mae[s_val]:14.4f}")

    # ══════════════════════════════════════════════════════════════
    # Censoring scenarios: [A] none / [B] Type I 20% / [C] Hybrid
    # Only Mode A K=1 (best configuration) for fair comparison
    # with rfl_inla.py scenarios.
    # ══════════════════════════════════════════════════════════════
    def apply_hybrid_censoring(Y, S, r_frac=0.80, T_cutoff=None):
        """Per stress-level hybrid censoring: T*_j = min(X_{r_j:n_j}, T_j)."""
        cens  = np.zeros(len(Y), bool)
        Y_obs = Y.copy()
        for s in np.unique(S):
            idx  = np.where(S == s)[0]
            n_j  = len(idx)
            r_j  = int(np.ceil(r_frac * n_j))
            x_r  = np.sort(Y[idx])[min(r_j - 1, n_j - 1)]
            T_st = min(x_r, T_cutoff) if T_cutoff is not None else x_r
            mask = Y[idx] > T_st
            cens[idx[mask]]  = True
            Y_obs[idx[mask]] = T_st
        return Y_obs, cens

    print("\n\n" + "=" * 65)
    print("CENSORING COMPARISON — Mode A K=1 (sig>=0.15)")
    print("Mirrors rfl_inla.py scenarios [A][B][C]")
    print("=" * 65)

    cens_scenarios = {}

    # [A] No censoring
    cens_scenarios['A'] = (Y_r.copy(), S_r, np.zeros(n, bool), 'No censoring')

    # [B] Type I — censor at 80th percentile per stress (~20% censored)
    Y_B   = Y_r.copy()
    cens_B = np.zeros(n, bool)
    for s in stresses:
        idx = np.where(S_r == s)[0]
        T_j = np.quantile(Y_r[idx], 0.80)
        mask = Y_r[idx] > T_j
        cens_B[idx[mask]] = True
        Y_B[idx[mask]]    = T_j
    cens_scenarios['B'] = (Y_B, S_r, cens_B, f'Type I (80th pct, {cens_B.mean():.1%} censored)')

    # [C] Hybrid I-II: r=80%, T = global 75th percentile
    T_hyb = np.quantile(Y_r, 0.75)
    Y_C, cens_C = apply_hybrid_censoring(Y_r, S_r, r_frac=0.80, T_cutoff=T_hyb)
    cens_scenarios['C'] = (Y_C, S_r, cens_C,
                           f'Hybrid I-II (r=80%, T=75th, {cens_C.mean():.1%} censored)')

    print(f"\n  {'Scenario':<42} {'n_obs':>5} {'n_cens':>6} {'ASSE':>6}  "
          f"{'log-lik':>9}  {'b0':>8}  {'b1':>8}  {'sig':>6}  {'a':>6}")
    print(f"  {'-'*42}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")

    # SEV+INLA reference row (from rfl_inla.py [A] result — no censoring)
    print(f"  {'[ref] SEV+INLA [A] no censoring':<42} {'75':>5}  {'0':>6}  {'5.76':>6}  "
          f"{'-72.880':>9}  {'-9.316':>8}  {'-8.522':>8}  {'0.190':>6}  {'n/a':>6}")

    scen_results = {}
    for label, (Y_sc, S_sc, cens_sc, desc) in cens_scenarios.items():
        print(f"\n  [{label}] {desc}")
        r = em_fit(Y_sc, S_sc, K=1, n_iter=60, verbose=False,
                   fix_sig=False, fix_a=False, cens=cens_sc)
        scen_results[label] = r
        n_obs_sc = (~cens_sc).sum()
        n_cen_sc = cens_sc.sum()
        print(f"  [{label}]  {'Burr+EM-GMM Mode A K=1':<40} {n_obs_sc:>5}  {n_cen_sc:>6}  "
              f"{r['asse']:6.2f}  {r['loglik']:9.4f}  {r['b0']:8.4f}  {r['b1']:8.4f}  "
              f"{r['sig']:6.4f}  {r['a']:6.3f}")

    # Summary table
    print("\n" + "─" * 65)
    print("  Summary: ASSE on uncensored observations (lower=better)")
    print("─" * 65)
    print(f"  {'Scenario':<35} {'n_obs':>5}  {'ASSE':>6}  {'sig':>6}  {'a':>6}")
    for label, (_, _, cens_sc, desc) in cens_scenarios.items():
        r = scen_results[label]
        print(f"  [{label}] {desc[:33]:<33} {(~cens_sc).sum():>5}  "
              f"{r['asse']:6.2f}  {r['sig']:6.4f}  {r['a']:6.3f}")
    print(f"\n  Normal+INLA [A] ASSE=8.63 / [B] ASSE=? / [C] crashed (b1=-20.4)")
    print(f"  SEV+INLA    [A] ASSE=5.76 / [B][C] not yet run")
