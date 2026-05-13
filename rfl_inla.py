"""
RFL — INLA-style Multiple Laplace Approximation + Heuristic Optimizer
======================================================================
Model
  g(Δ)     ~ LogNormal(μ_Δ, σ_Δ²)          [parametric; integral needed]
  f(Y|Δ,S) ~ N(β₀ + β₁ log(S−Δ), σ²)      [Roy's Normal version]

Integration strategy (two-level, INLA philosophy)
  Inner  : per-observation ∫h_i(Δ)dΔ via Multiple Laplace
           = Gauss-Hermite (N_GH nodes) centred at the posterior mode of Δ_i
  Outer  : hyperparameter θ=(β₀,β₁,σ,μ_Δ,σ_Δ) optimised by
           grid random-search → dual_annealing (SA) → Nelder-Mead

Censoring
  Type I  : fixed cutoff T_j per stress level
  Hybrid  : T* = min(X_{r:n,j}, T_j) per stress level

References
  Pascual & Meeker (1999) Technometrics 41(4)
  Rue, Martino & Chopin (2009) JRSS-B 71(2)  [INLA paper]
  Gauss-Hermite: numpy.polynomial.hermite_e.hermegauss (probabilist's form)
"""
import numpy as np
from scipy.stats import norm, lognorm
from scipy.optimize import minimize_scalar, minimize, dual_annealing
import warnings; warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 1.  Gauss-Hermite nodes/weights (probabilist's form)
#     ∫ f(x) e^{-x²/2} dx ≈ Σ_j w_j f(x_j)
# ═══════════════════════════════════════════════════════════════════
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)
# Precompute exp(x_j²/2) — used to cancel the Gaussian kernel in IS-GH
_GHE = np.exp(_GHX ** 2 / 2)


# ═══════════════════════════════════════════════════════════════════
# 2.  Per-observation integrand and Multiple Laplace integral
# ═══════════════════════════════════════════════════════════════════

def _log_h(d, y, s, is_cens, b0, b1, sig, mu_d, sig_d):
    """
    log h_i(Δ) = log f(y_i | Δ, S_i) + log g(Δ)
    Handles both uncensored (is_cens=False) and Type I right-censored.
    Returns -1e30 outside the valid domain (0, S_i).
    """
    if d <= 0 or d >= s - 1e-8:
        return -1e30
    mu_y = b0 + b1 * np.log(s - d)
    # Likelihood contribution
    if is_cens:
        lf = norm.logsf((y - mu_y) / sig)     # log P(Y > c | Δ)
    else:
        lf = norm.logpdf(y, mu_y, sig)
    # Prior / mixing distribution
    lg = lognorm.logpdf(d, s=sig_d, scale=np.exp(mu_d))
    return lf + lg


def _multi_laplace(y, s, is_cens, b0, b1, sig, mu_d, sig_d):
    """
    Compute log L_i = log ∫₀ˢ h_i(Δ) dΔ via Multiple Laplace.

    Algorithm (importance-weighted Gauss-Hermite):
      1. Find mode  Δ̂ = argmax log h_i(Δ)
      2. Curvature  κ = -h''(Δ̂),  σ̃ = 1/√κ  (Laplace scale)
      3. GH nodes   Δ_j = Δ̂ + σ̃ x_j
      4. L_i ≈ σ̃ Σ_j w_j exp(x_j²/2) h_i(Δ_j)
         [the exp(x_j²/2) factor cancels the Gaussian kernel of the IS proposal]

    Single Laplace = first-order approximation of this.
    """
    # ── mode ────────────────────────────────────────────────────────
    res = minimize_scalar(
        lambda d: -_log_h(d, y, s, is_cens, b0, b1, sig, mu_d, sig_d),
        bounds=(1e-6, s - 1e-4),
        method='bounded',
    )
    d_hat = res.x
    lh0   = _log_h(d_hat, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    if lh0 < -1e25:          # mode outside valid region → degenerate
        return -1e30

    # ── curvature via central finite differences ─────────────────────
    eps   = max(d_hat * 0.005, 5e-6)
    lhp   = _log_h(d_hat + eps, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    lhm   = _log_h(d_hat - eps, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    kappa = max(-(lhp - 2 * lh0 + lhm) / eps ** 2, 1e-6)
    sig_t = 1.0 / np.sqrt(kappa)

    # ── GH quadrature ───────────────────────────────────────────────
    d_pts  = d_hat + sig_t * _GHX          # (N_GH,) quadrature points in Δ
    lh_pts = np.array([
        _log_h(dj, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
        for dj in d_pts
    ])
    # L_i = σ̃ Σ_j w_j exp(x_j²/2) exp(log h(Δ_j))
    # Work in log-space to avoid overflow: factor out max(lh_pts)
    log_terms  = lh_pts + _GHX ** 2 / 2
    log_shift  = log_terms.max()
    L_i        = sig_t * np.sum(_GHW * np.exp(log_terms - log_shift)) * np.exp(log_shift)
    return np.log(max(L_i, 1e-300))


def loglik(theta, Y, S, cens):
    """
    Total log-likelihood  ℓ(θ) = Σ_i log L_i
    theta = (b0, b1, log_sig, mu_d, log_sig_d)   [log-transforms for positivity]
    """
    b0, b1, log_sig, mu_d, log_sig_d = theta
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)
    if sig < 0.01 or sig_d < 0.01:
        return -1e30
    return sum(
        _multi_laplace(Y[i], S[i], bool(cens[i]), b0, b1, sig, mu_d, sig_d)
        for i in range(len(Y))
    )


# ═══════════════════════════════════════════════════════════════════
# 3.  Hybrid Type I-II censoring generator
# ═══════════════════════════════════════════════════════════════════

def apply_hybrid_censoring(Y, S, r_frac=0.80, T_cutoff=None):
    """
    Per stress-level hybrid censoring: T*_j = min(X_{r_j:n_j}, T_j)

    Parameters
    ----------
    r_frac   : proportion of specimens defining the Type II threshold
               r_j = ceil(r_frac * n_j)
    T_cutoff : global Type I cutoff (on log-life scale); None = no Type I component

    Returns
    -------
    Y_obs  : observed (possibly right-censored) log-lives
    cens   : boolean array, True = right-censored
    """
    Y_obs = Y.copy()
    cens  = np.zeros(len(Y), bool)
    for s in np.unique(S):
        idx  = np.where(S == s)[0]
        n_j  = len(idx)
        r_j  = int(np.ceil(r_frac * n_j))
        # Type II threshold: r-th order statistic
        x_r  = np.sort(Y[idx])[min(r_j - 1, n_j - 1)]
        T_star = min(x_r, T_cutoff) if T_cutoff is not None else x_r
        mask = Y[idx] > T_star
        cens[idx[mask]]  = True
        Y_obs[idx[mask]] = T_star
    return Y_obs, cens


# ═══════════════════════════════════════════════════════════════════
# 4.  Heuristic optimizer (3-stage)
# ═══════════════════════════════════════════════════════════════════
# Parameter order: (b0, b1, log_sig, mu_d, log_sig_d)
_BOUNDS = [(-20., -2.), (-20., -2.),
           (np.log(0.05), np.log(3.0)),
           (-3.0, 1.0),
           (np.log(0.05), np.log(1.5))]


def heuristic_optimize(Y, S, cens, n_grid=40, sa_maxiter=800, seed=2025, verbose=True):
    """
    Stage 1 — Random grid:  n_grid random θ samples → pick best
    Stage 2 — Dual annealing (SA):  global heuristic search from best grid point
    Stage 3 — Nelder-Mead:  local polish of SA result

    Returns
    -------
    theta_hat : np.ndarray of shape (5,)  [b0, b1, log_sig, mu_d, log_sig_d]
    ll_hat    : float, maximum log-likelihood
    """
    rng = np.random.default_rng(seed)

    # ── Stage 1: random grid ────────────────────────────────────────
    best_ll = -np.inf; best_t = None
    for _ in range(n_grid):
        t = [rng.uniform(lo, hi) for lo, hi in _BOUNDS]
        try:
            ll = loglik(t, Y, S, cens)
            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll; best_t = t[:]
        except Exception:
            pass
    if verbose:
        print(f"  [Grid  ] best ll = {best_ll:.3f}")

    # ── Stage 2: dual_annealing (SA + local) ────────────────────────
    # dual_annealing combines simulated annealing with a gradient-based
    # local minimiser at each accepted point — "heuristic learning" in
    # the sense that it adapts the search temperature to the landscape.
    res_sa = dual_annealing(
        lambda t: -loglik(t, Y, S, cens),
        bounds=_BOUNDS,
        x0=best_t,
        seed=seed,
        maxiter=sa_maxiter,
        minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': _BOUNDS},
        no_local_search=False,
    )
    if verbose:
        print(f"  [SA    ] best ll = {-res_sa.fun:.3f}  (iters={res_sa.nit})")

    # ── Stage 3: Nelder-Mead polish ─────────────────────────────────
    res_nm = minimize(
        lambda t: -loglik(t, Y, S, cens),
        x0=res_sa.x,
        method='Nelder-Mead',
        options={'maxiter': 15000, 'xatol': 1e-7, 'fatol': 1e-7},
    )
    if verbose:
        print(f"  [NM    ] best ll = {-res_nm.fun:.3f}")
    return res_nm.x, float(-res_nm.fun)


# ═══════════════════════════════════════════════════════════════════
# 5.  Standard error via numerical Hessian of ℓ(θ) w.r.t. ψ=(b0,b1,σ)
# ═══════════════════════════════════════════════════════════════════

def se_hessian(theta_hat, Y, S, cens, h_frac=0.04):
    """
    SE for ψ = (b0, b1, σ) via central-difference Hessian of ℓ.
    Nuisance (μ_Δ, σ_Δ) held fixed at their MLE.
    For profile SE, replace loglik with a profile over (μ_Δ, σ_Δ) — here we
    use the observed-information approximation (fast, adequate for comparison).
    """
    b0, b1, ls, mu_d, lsd = theta_hat
    psi   = np.array([b0, b1, ls])

    def ll3(p):
        return loglik([p[0], p[1], p[2], mu_d, lsd], Y, S, cens)

    h   = np.abs(psi) * h_frac + 1e-3
    ll0 = ll3(psi)
    H   = np.zeros((3, 3))
    for i in range(3):
        ei       = np.zeros(3); ei[i] = h[i]
        H[i, i]  = (ll3(psi + ei) - 2 * ll0 + ll3(psi - ei)) / h[i] ** 2
        for j in range(i + 1, 3):
            ej       = np.zeros(3); ej[j] = h[j]
            H[i, j] = H[j, i] = (
                ll3(psi + ei + ej) - ll3(psi + ei - ej)
                - ll3(psi - ei + ej) + ll3(psi - ei - ej)
            ) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(-H)
        se  = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    return se, H


# ═══════════════════════════════════════════════════════════════════
# 6.  Single Laplace (for comparison with Multiple Laplace)
# ═══════════════════════════════════════════════════════════════════

def loglik_single_laplace(theta, Y, S, cens):
    """P&M-style single Laplace: L_i ≈ h_i(Δ̂) √(2π/κ)."""
    b0, b1, log_sig, mu_d, log_sig_d = theta
    sig = np.exp(log_sig); sig_d = np.exp(log_sig_d)
    ll = 0.
    for i in range(len(Y)):
        y, s, c = Y[i], S[i], bool(cens[i])
        def log_h(d): return _log_h(d, y, s, c, b0, b1, sig, mu_d, sig_d)
        res  = minimize_scalar(lambda d: -log_h(d), bounds=(1e-6, s - 1e-4), method='bounded')
        d_hat = res.x; lh0 = log_h(d_hat)
        eps   = max(d_hat * 0.005, 5e-6)
        kappa = max(-(log_h(d_hat + eps) - 2*lh0 + log_h(d_hat - eps)) / eps**2, 1e-6)
        ll   += lh0 + 0.5 * np.log(2 * np.pi / kappa)
    return ll


# ═══════════════════════════════════════════════════════════════════
# 7.  Real data — Pascual & Meeker (1999), n=75
# ═══════════════════════════════════════════════════════════════════
_raw = {
    0.675: [102.95, 280.32, 339.83, 366.9,  485.62,  658.96,  896.33,
            1241.76, 1250.2, 1329.78, 1399.83, 1459.14, 3249.82, 11748.1, 11748.1],
    0.75:  [6.71, 9.93, 12.6, 15.58, 16.19, 17.28, 18.62,
            20.3, 24.9, 26.26, 27.94, 36.35, 48.42, 50.09, 67.34],
    0.825: [1.246, 1.258, 1.46, 1.492, 2.4, 2.41, 2.59,
            2.903, 3.33, 3.59, 3.847, 4.11, 4.82, 5.56, 5.598],
    0.9:   [0.201, 0.216, 0.226, 0.252, 0.257, 0.295, 0.311,
            0.342, 0.356, 0.451, 0.457, 0.509, 0.54, 0.68, 1.129],
    0.95:  [0.037, 0.072, 0.074, 0.076, 0.083, 0.085, 0.105,
            0.109, 0.12, 0.123, 0.143, 0.203, 0.206, 0.217, 0.257],
}
S_all, Y_all = [], []
for s, ys in _raw.items():
    for y in ys:
        S_all.append(s); Y_all.append(np.log(y))
S_r = np.array(S_all)
Y_r = np.array(Y_all)
cens_none = np.zeros(len(Y_r), bool)


# ═══════════════════════════════════════════════════════════════════
# 8.  Main experiments
# ═══════════════════════════════════════════════════════════════════
def _print_result(label, theta_hat, ll, se=None):
    b0, b1, ls, mu_d, lsd = theta_hat
    sig = np.exp(ls); sig_d = np.exp(lsd)
    E_delta = np.exp(mu_d + sig_d**2 / 2)
    print(f"\n  MLE  b0={b0:.4f}  b1={b1:.4f}  σ={sig:.4f}")
    print(f"       μ_Δ={mu_d:.4f}  σ_Δ={sig_d:.4f}  E[Δ]={E_delta:.4f}")
    print(f"       ll={ll:.4f}")
    if se is not None:
        print(f"  SE   b0={se[0]:.4f}  b1={se[1]:.4f}  log_σ={se[2]:.4f}")


if __name__ == '__main__':
    print("=" * 65)
    print("RFL  INLA-style Multiple Laplace  (LogNormal g + Normal f)")
    print(f"GH nodes = {N_GH}   n = {len(Y_r)}")
    print("=" * 65)

    # ── A: Full data, no censoring ───────────────────────────────────
    print("\n[A] No censoring  (n=75)")
    th_A, ll_A = heuristic_optimize(Y_r, S_r, cens_none)
    se_A, _    = se_hessian(th_A, Y_r, S_r, cens_none)
    _print_result("A", th_A, ll_A, se_A)

    # ── B: Type I censoring (~20% per stress level) ──────────────────
    print("\n" + "─" * 65)
    print("\n[B] Type I censoring  (T = 80th pct per stress, ~20% censored)")
    Y_B = Y_r.copy(); cens_B = np.zeros(len(Y_r), bool)
    for s in np.unique(S_r):
        idx = np.where(S_r == s)[0]
        T_j = np.quantile(Y_r[idx], 0.80)
        mask = Y_r[idx] > T_j
        cens_B[idx[mask]] = True; Y_B[idx[mask]] = T_j
    print(f"  Censoring rate: {cens_B.mean():.1%}")
    th_B, ll_B = heuristic_optimize(Y_B, S_r, cens_B)
    se_B, _    = se_hessian(th_B, Y_B, S_r, cens_B)
    _print_result("B", th_B, ll_B, se_B)

    # ── C: Hybrid Type I-II censoring ───────────────────────────────
    print("\n" + "─" * 65)
    print("\n[C] Hybrid Type I-II  (r=80%, T = global 75th pct)")
    T_hyb = np.quantile(Y_r, 0.75)
    Y_C, cens_C = apply_hybrid_censoring(Y_r, S_r, r_frac=0.80, T_cutoff=T_hyb)
    print(f"  Censoring rate: {cens_C.mean():.1%}")
    th_C, ll_C = heuristic_optimize(Y_C, S_r, cens_C)
    se_C, _    = se_hessian(th_C, Y_C, S_r, cens_C)
    _print_result("C", th_C, ll_C, se_C)

    # ── D: Single vs Multiple Laplace (full data) ────────────────────
    print("\n" + "─" * 65)
    print("\n[D] Single vs Multiple Laplace comparison  (full data, MLE from A)")
    ll_single = loglik_single_laplace(th_A, Y_r, S_r, cens_none)
    ll_multi  = loglik(th_A, Y_r, S_r, cens_none)
    print(f"  Single Laplace  ll = {ll_single:.5f}")
    print(f"  Multi  Laplace  ll = {ll_multi:.5f}")
    print(f"  diff = {ll_multi - ll_single:+.5f}  "
          f"({'multi > single' if ll_multi > ll_single else 'single >= multi'})")

    # ── E: Per-observation mode diagnostics ─────────────────────────
    print("\n" + "─" * 65)
    print("\n[E] Per-observation Laplace mode Δ̂_i  (full data, first 5 per stress)")
    b0, b1, ls, mu_d, lsd = th_A
    sig = np.exp(ls); sig_d = np.exp(lsd)
    print(f"  {'S_i':>6}  {'Y_i':>7}  {'D_hat':>8}  {'sig_t':>8}  {'logL_i':>10}")
    printed = {}
    for i in range(len(Y_r)):
        s = S_r[i]; y = Y_r[i]
        if printed.get(s, 0) >= 3:
            continue
        printed[s] = printed.get(s, 0) + 1
        # recompute mode + curvature
        res = minimize_scalar(
            lambda d: -_log_h(d, y, s, False, b0, b1, sig, mu_d, sig_d),
            bounds=(1e-6, s - 1e-4), method='bounded')
        d_hat = res.x; lh0 = _log_h(d_hat, y, s, False, b0, b1, sig, mu_d, sig_d)
        eps = max(d_hat*0.005, 5e-6)
        kappa = max(-(_log_h(d_hat+eps,y,s,False,b0,b1,sig,mu_d,sig_d)
                      - 2*lh0 +
                      _log_h(d_hat-eps,y,s,False,b0,b1,sig,mu_d,sig_d)) / eps**2, 1e-6)
        logLi = _multi_laplace(y, s, False, b0, b1, sig, mu_d, sig_d)
        print(f"  {s:6.3f}  {y:7.4f}  {d_hat:8.5f}  {1/np.sqrt(kappa):8.5f}  {logLi:10.5f}")

    print("\nDone.")
