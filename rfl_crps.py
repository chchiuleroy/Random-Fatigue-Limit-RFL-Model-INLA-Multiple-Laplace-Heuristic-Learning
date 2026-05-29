"""
rfl_crps.py — CRPS-based estimation for RFL model
===================================================

Minimises marginal CRPS instead of log-likelihood:

  CRPS(N(mu,sig), y) = sig { z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) }
                       where z = (y-mu)/sig

  CRPS_marg(y; S, theta) = E_Delta[ CRPS(N(b0+b1*ln(S-Delta), sig_e), y) ]
                         ≈ (1/sqrt(pi)) sum_k w_k CRPS(N(mu_k, sig_e), y)   [40-pt GH]

Why CRPS:
  MLE finds sig_e=0.131 (tight) → rank-ASSE=12.85  (maximises precision)
  CRPS finds sig_e≈?    (spread) → rank-ASSE≈?      (maximises calibration)
  Direct ASSE opt: sig_e=0.258  → rank-ASSE=12.24   (minimises ASSE directly)
  CRPS should land between MLE and direct ASSE opt.

Censoring schema:
  - Uncensored (delta=1): contributes CRPS_marg(y; S, theta)
  - Right-censored (delta=0) at c: contributes -log(1-F_marg(c; S, theta))
    (proper likelihood term; handles runout specimens)

Result: single fitting pass handles mixed censoring, gives ASSE-optimised theta.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from numpy.polynomial.hermite import hermgauss
from joblib import Parallel, delayed

# ── Data (same as rfl_chiu.py) ────────────────────────────────────────────────
_raw = {
    0.675: [102.95, 280.32, 339.83, 366.9, 485.62, 658.96, 896.33,
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

S_all, Y_all, grp_all = [], [], []
for j, (s, ys) in enumerate(_raw.items()):
    for y in ys:
        S_all.append(s); Y_all.append(np.log(y)); grp_all.append(j)
S_all   = np.array(S_all)
Y_all   = np.array(Y_all)
grp_all = np.array(grp_all)
N_OBS   = len(Y_all)
DELTA   = np.ones(N_OBS, dtype=int)  # all uncensored in P&M dataset

# GH nodes
N_GH        = 40
_GHX, _GHW = hermgauss(N_GH)

# Pre-sorted groups for rank-ASSE evaluation
_groups = []
for s, ys in _raw.items():
    om_s = np.sort(np.log(ys)); n = len(om_s)
    ps   = [(i + 0.5) / n for i in range(n)]
    _groups.append((s, om_s, ps))


# ── GH helper: LogNormal quadrature nodes ─────────────────────────────────────
def _gh_lognormal_nodes(mu_D, sig_D):
    """Return (d_v, w_v, w_norm) for GH over LogNormal(mu_D, sig_D^2)."""
    ln_d = mu_D + np.sqrt(2) * sig_D * _GHX
    d_v  = np.exp(ln_d)
    return d_v, _GHW


# ── CRPS helper primitives ────────────────────────────────────────────────────
def _abs_mean(y, mu, sig):
    """E_{X~N(mu,sig)}[|X-y|] = sig*(z*(2Phi(z)-1) + 2*phi(z))."""
    z = (y - mu) / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z))

def _abs_diff_mean(m1, m2, sig_e):
    """E[|X-X'|] where X~N(m1,sig_e), X'~N(m2,sig_e), so X-X'~N(d,2*sig_e^2)."""
    d = m1 - m2
    se2 = np.sqrt(2) * sig_e
    return 2*sig_e/np.sqrt(np.pi)*np.exp(-d**2/(4*sig_e**2)) + d*(2*norm.cdf(d/se2)-1)


# ── Correct marginal CRPS via energy score ────────────────────────────────────
def marg_crps(y, S, b0, b1, sig_e, mu_D, sig_D):
    """
    CRPS(F_marg, y) using the energy score decomposition:
      CRPS_marg(y) = E_Delta[|X_Delta - y|] - (1/2) E_{Delta,Delta'}[|X_Delta - X_Delta'|]
    where X_Delta ~ N(b0+b1*ln(S-Delta), sig_e).

    Both expectations evaluated via 40-pt Gauss-Hermite over LogNormal(mu_D, sig_D^2).
    This avoids the Jensen gap from averaging conditional CRPSes directly.
    """
    d_v, w_v = _gh_lognormal_nodes(mu_D, sig_D)
    valid = d_v < S - 1e-9
    if not valid.any():
        return float('inf')
    d_v = d_v[valid]; w_v = w_v[valid]
    mu_v   = b0 + b1 * np.log(S - d_v)
    wn     = w_v / np.sqrt(np.pi)   # normalised GH weights, sum≈1

    # Term 1: E_Delta[|X - y|]
    term1 = float(np.dot(wn, [_abs_mean(y, mu, sig_e) for mu in mu_v]))

    # Term 2: (1/2) E_{Delta,Delta'}[|X_Delta - X_Delta'|]
    # Double GH sum — O(K^2) but K≤40 so fast
    mu_mat   = mu_v[:, None] - mu_v[None, :]          # (K,K) pairwise diff
    h_mat    = (_abs_diff_mean_vectorised(mu_mat, sig_e))
    w2_mat   = wn[:, None] * wn[None, :]              # outer product of weights
    term2    = 0.5 * float(np.sum(w2_mat * h_mat))

    return term1 - term2

def _abs_diff_mean_vectorised(diff, sig_e):
    """Vectorised version of _abs_diff_mean for a matrix of diffs."""
    se2 = np.sqrt(2) * sig_e
    return (2*sig_e/np.sqrt(np.pi)*np.exp(-diff**2/(4*sig_e**2))
            + diff*(2*norm.cdf(diff/se2)-1))


# ── Marginal CDF (for censored likelihood term & rank-ASSE eval) ──────────────
def marg_cdf(omega, S, b0, b1, sig_e, mu_D, sig_D):
    """F_marg(omega|S;theta) = E_Delta[Phi((omega-mu(S,Delta))/sig_e)]."""
    d_v, w_v = _gh_lognormal_nodes(mu_D, sig_D)
    valid = d_v < S - 1e-9
    if not valid.any():
        return 0.5
    d_v = d_v[valid]; w_v = w_v[valid]
    mu_v = b0 + b1 * np.log(S - d_v)
    return float(np.dot(w_v, norm.cdf((omega - mu_v) / sig_e)) / np.sqrt(np.pi))


# ── Total fitting loss ─────────────────────────────────────────────────────────
def total_loss(b0, b1, sig_e, mu_D, sig_D, Y=Y_all, S=S_all, delta=DELTA):
    """
    Fitting loss supporting mixed censoring:
      uncensored (delta=1): CRPS_marg(y; S, theta)  [energy score, correct marginal]
      right-censored (delta=0) at c: -log(1-F_marg(c; S, theta))
    marg_crps already uses vectorised (K×K) matrix ops — no extra parallelism needed.
    """
    loss = 0.0
    for y, s, d in zip(Y, S, delta):
        if d == 1:
            v = marg_crps(y, s, b0, b1, sig_e, mu_D, sig_D)
            loss += v if np.isfinite(v) else 1e10
        else:
            F = marg_cdf(y, s, b0, b1, sig_e, mu_D, sig_D)
            loss -= np.log(max(1.0 - F, 1e-10))
    return loss if np.isfinite(loss) else 1e10


# ── Vectorised marginal CDF over omega grid ────────────────────────────────────
def _marg_cdf_vec(omega_arr, S, b0, b1, sig_e, mu_D, sig_D):
    """F_marg for a vector of omegas at fixed S — one (M×K) matrix multiply."""
    ln_d = mu_D + np.sqrt(2) * sig_D * _GHX
    d_v  = np.exp(ln_d)
    valid = d_v < S - 1e-9
    if not valid.any():
        return np.full(len(omega_arr), 0.5)
    d_v = d_v[valid]; w_v = _GHW[valid]
    mu_v  = b0 + b1 * np.log(S - d_v)
    z_mat = (omega_arr[:, None] - mu_v[None, :]) / sig_e
    return norm.cdf(z_mat) @ w_v / np.sqrt(np.pi)


# ── Rank-ASSE via grid + interpolation (fast) ─────────────────────────────────
def rank_asse(b0, b1, sig_e, mu_D, sig_D, grid_pts=400):
    """rank-ASSE: 5 matrix ops (one per group) + np.interp, no brentq."""
    asse = 0.0
    for s, om_s, ps in _groups:
        lo = om_s[0] - 3.0; hi = om_s[-1] + 3.0
        grid  = np.linspace(lo, hi, grid_pts)
        F_vec = _marg_cdf_vec(grid, s, b0, b1, sig_e, mu_D, sig_D)
        for om, p in zip(om_s, ps):
            asse += abs(om - float(np.interp(p, F_vec, grid)))
    return asse


# ── CRPS optimisation ─────────────────────────────────────────────────────────
def run_crps_opt(theta0=None, verbose=True):
    """
    Minimise total_loss() over theta=(b0,b1,log_sig_e,mu_D,log_sig_D).
    For uncensored data this is pure CRPS; for censored data it is
    CRPS (uncensored) + negative log-survival (censored).

    Starting point: Chiu's EIV parameters.
    """
    if theta0 is None:
        theta0 = [-9.074687, -7.602654,
                  np.log(np.sqrt(0.0312110845352731)),  # log(0.1767)
                  -0.5952965, np.log(0.0345517)]

    def obj(t):
        b0, b1, lse, mu_D, lsd = t
        sig_e = np.exp(lse); sig_D = np.exp(lsd)
        if sig_e < 0.01 or sig_D < 0.001:
            return 1e10
        return total_loss(b0, b1, sig_e, mu_D, sig_D)

    if verbose:
        b0, b1, lse, mu_D, lsd = theta0
        sig_e0 = np.exp(lse)
        print(f"=== CRPS-based optimisation (Normal model, 40-pt GH) ===")
        print(f"  Start: CRPS-loss = {obj(theta0):.4f}  sig_e={sig_e0:.4f}")

    res = minimize(obj, theta0, method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-7,
                            'maxiter': 15000, 'adaptive': True})
    b0, b1, lse, mu_D, lsd = res.x
    sig_e = np.exp(lse); sig_D = np.exp(lsd)

    if verbose:
        print(f"  CRPS-loss at optimum = {res.fun:.4f}")
        print(f"  b0={b0:.4f}  b1={b1:.4f}  sig_e={sig_e:.4f}")
        print(f"  mu_D={mu_D:.4f}  sig_D={sig_D:.5f}")
        print(f"  Evaluating rank-ASSE (slow)...")

    asse_r = rank_asse(b0, b1, sig_e, mu_D, sig_D)

    result = dict(b0=b0, b1=b1, sig_e=sig_e, mu_D=mu_D, sig_D=sig_D,
                  crps_loss=res.fun, asse_rank=asse_r)

    if verbose:
        print(f"  rank-ASSE = {asse_r:.4f}")

    return result


# ── Profile over sig_e (fix other params from Chiu, sweep sig_e) ──────────────
def run_profile_sige(verbose=True):
    """
    Quick diagnostic: keep (b0,b1,mu_D,sig_D) fixed at Chiu's values,
    sweep log(sig_e) to find the sig_e that minimises rank-ASSE.
    Shows the ASSE surface as a function of sig_e only.
    """
    from scipy.optimize import minimize_scalar

    b0    = -9.074687
    b1    = -7.602654
    mu_D  = -0.5952965
    sig_D =  0.0345517

    if verbose:
        print(f"\n=== Profile sig_e → rank-ASSE (Chiu params fixed) ===")
        for sig_e in [0.10, 0.15, 0.177, 0.20, 0.25, 0.30, 0.35, 0.40]:
            a = rank_asse(b0, b1, sig_e, mu_D, sig_D)
            print(f"  sig_e={sig_e:.3f}  rank-ASSE={a:.4f}")

    def neg(lse):
        sig_e = np.exp(lse)
        if sig_e < 0.05 or sig_e > 1.0:
            return 1e10
        return rank_asse(b0, b1, sig_e, mu_D, sig_D)

    res = minimize_scalar(neg, bounds=(np.log(0.05), np.log(1.0)), method='bounded')
    sig_e_opt = np.exp(res.x)
    asse_opt  = res.fun

    if verbose:
        print(f"  Profile optimum: sig_e={sig_e_opt:.4f}  rank-ASSE={asse_opt:.4f}")

    return sig_e_opt, asse_opt


# ── Kaplan-Meier plotting positions for censored data ─────────────────────────
def km_plotting_positions(failures, runouts):
    """
    Kaplan-Meier plotting positions for uncensored failures given runouts.

    failures : sorted array of uncensored log-failure times (ln Y)
    runouts  : array of log-censoring times (ln C, where true Y > C)

    Returns list of (omega, p) where p = 1 - KM(omega), the estimated CDF
    at each failure time. For runouts=[], reduces to (i-0.5)/n.
    """
    n_f = len(failures); n_c = len(runouts)
    n   = n_f + n_c

    # Build event list: (time, delta=1 for failure, 0 for runout)
    events = sorted([(t, 1) for t in failures] + [(t, 0) for t in runouts])

    S = 1.0; at_risk = n; result = []
    for t, d in events:
        if d == 1:
            S *= (at_risk - 1) / at_risk
            # Hazen correction: (i-0.5)/n equivalent under censoring
            # Raw KM gives i/n; subtract 0.5/n to align with (i-0.5)/n for uncensored
            p_hazen = max(1e-6, (1.0 - S) - 0.5 / n)
            result.append((t, p_hazen))
        at_risk -= 1
    return result  # list of (omega, p) for uncensored obs only


# ── Rank-ASSE with KM plotting positions ──────────────────────────────────────
def km_rank_asse(b0, b1, sig_e, mu_D, sig_D, groups_km, grid_pts=400):
    """ASSE with KM plotting positions — vectorised grid + interp."""
    asse = 0.0
    for s, om_failures, ps in groups_km:
        if not om_failures: continue
        lo = om_failures[0] - 3.0; hi = om_failures[-1] + 3.0
        grid  = np.linspace(lo, hi, grid_pts)
        F_vec = _marg_cdf_vec(grid, s, b0, b1, sig_e, mu_D, sig_D)
        for om, p in zip(om_failures, ps):
            asse += abs(om - float(np.interp(p, F_vec, grid)))
    return asse


def run_km_asse_opt(groups_km, theta0=None, label="KM-ASSE", verbose=True):
    """
    Directly minimise rank-ASSE with KM plotting positions (handles censored data).
    For all-uncensored data this is identical to run_rank_asse_opt in rfl_chiu.py.
    """
    if theta0 is None:
        theta0 = [-9.074687, -7.602654,
                  np.log(np.sqrt(0.0312110845352731)),
                  -0.5952965, np.log(0.0345517)]

    def obj(t):
        b0, b1, lse, mu_D, lsd = t
        sig_e = np.exp(lse); sig_D = np.exp(lsd)
        if sig_e < 0.01 or sig_D < 0.001: return 1e10
        try: return km_rank_asse(b0, b1, sig_e, mu_D, sig_D, groups_km)
        except: return 1e10

    if verbose:
        print(f"\n=== {label} optimisation ===")
        print(f"  Start: ASSE = {obj(theta0):.4f}")

    res = minimize(obj, theta0, method='Nelder-Mead',
                   options={'xatol':1e-7,'fatol':1e-7,'maxiter':10000,'adaptive':True})
    b0, b1, lse, mu_D, lsd = res.x
    sig_e = np.exp(lse); sig_D = np.exp(lsd)

    # Standard rank_asse at optimal params (for comparison with other methods)
    std_asse = rank_asse(b0, b1, sig_e, mu_D, sig_D)
    result = dict(b0=b0, b1=b1, sig_e=sig_e, mu_D=mu_D, sig_D=sig_D,
                  km_asse=res.fun, asse=std_asse)
    if verbose:
        print(f"  KM-ASSE (loss) = {res.fun:.4f}")
        print(f"  rank-ASSE (std, (i-0.5)/n) = {std_asse:.4f}")
        print(f"  b0={b0:.4f}  b1={b1:.4f}  sig_e={sig_e:.4f}")
        print(f"  mu_D={mu_D:.4f}  sig_D={sig_D:.5f}")
    return result


# ── Build KM groups from raw data (all-uncensored, no runouts) ─────────────────
_groups_km_full = []
for s, ys in _raw.items():
    failures = np.sort(np.log(ys))
    kmp = km_plotting_positions(failures, runouts=[])
    oms, ps = zip(*kmp)
    _groups_km_full.append((s, list(oms), list(ps)))


# ── Synthetic censoring demo ───────────────────────────────────────────────────
def synthetic_censored_demo(censor_pct=0.2, seed=42, verbose=True):
    """
    Drop censor_pct fraction of obs per group (simulate runouts at the largest values).
    Build KM-adjusted ASSE and optimise. Compare to all-uncensored result.
    """
    rng = np.random.default_rng(seed)
    groups_cens = []
    for s, ys in _raw.items():
        om_sorted = np.sort(np.log(ys)); n = len(om_sorted)
        n_cens = max(1, int(n * censor_pct))
        # censor the top n_cens obs (simulate runouts at max test time)
        failures = om_sorted[:-n_cens]
        runouts  = om_sorted[-n_cens:]  # these are RIGHT-CENSORED at their values
        kmp = km_plotting_positions(failures, runouts)
        if len(kmp) == 0:
            continue
        oms, ps = zip(*kmp)
        groups_cens.append((s, list(oms), list(ps)))

    if verbose:
        n_fail  = sum(len(g[1]) for g in groups_cens)
        n_total = sum(len(ys) for ys in _raw.values())
        print(f"\n=== Synthetic censoring demo ({censor_pct:.0%} runouts per group) ===")
        print(f"  {n_fail}/{n_total} failures observed")

    return run_km_asse_opt(groups_cens, label=f"KM-ASSE ({censor_pct:.0%} censored)", verbose=verbose)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    print("=" * 62)
    print("Profile sig_e (diagnostic — fast)")
    print("=" * 62)
    t0 = time.time()
    run_profile_sige(verbose=True)
    print(f"  [elapsed {time.time()-t0:.1f}s]")

    print()
    print("=" * 62)
    print("CRPS optimisation (all 5 params, uncensored data)")
    print("=" * 62)
    t0 = time.time()
    res_crps = run_crps_opt(verbose=True)
    print(f"  [elapsed {time.time()-t0:.1f}s]")

    print()
    print("=" * 62)
    print("KM-ASSE opt (all-uncensored — should match rfl_chiu Method B)")
    print("=" * 62)
    t0 = time.time()
    res_km_full = run_km_asse_opt(_groups_km_full, label="KM-ASSE (0% censored)", verbose=True)
    print(f"  [elapsed {time.time()-t0:.1f}s]")

    print()
    print("=" * 62)
    print("KM-ASSE opt (20% synthetic runouts per group)")
    print("=" * 62)
    t0 = time.time()
    res_km_cens = synthetic_censored_demo(censor_pct=0.20, verbose=True)
    print(f"  [elapsed {time.time()-t0:.1f}s]")

    print()
    print("=" * 62)
    print("Scoreboard — rank-ASSE (all uncensored data)")
    print("=" * 62)
    print(f"  Normal + direct rank-ASSE opt:  12.2370  (rfl_chiu.py Method B)")
    print(f"  Normal + KM-ASSE opt (0% cens): {res_km_full['asse']:.4f}  ← same method, KM wrapper")
    print(f"  Normal + CRPS opt:              {res_crps['asse_rank']:.4f}  ← CRPS proxy")
    print(f"  Chiu (2005) EIV:                12.4063")
    print(f"  SEV+NPMLE K=6:                  12.6100")
    print(f"  Normal+INLA (MLE):              12.8500")
    print()
    print(f"  KM-ASSE with 20% runouts:       {res_km_cens['asse']:.4f}  (fitted on {int(0.8*75)} obs)")
    print()
    print(f"  sig_e comparison:")
    print(f"    MLE (INLA):          ~0.131")
    print(f"    Chiu EIV:             0.177")
    print(f"    CRPS opt:             {res_crps['sig_e']:.3f}")
    print(f"    KM-ASSE 0%:           {res_km_full['sig_e']:.3f}")
    print(f"    KM-ASSE 20%:          {res_km_cens['sig_e']:.3f}")
    print(f"    Direct rank-ASSE:     0.258")
    print()
    print("Censoring schema summary:")
    print("  CRPS method: total_loss(delta=[1,0,...]) — censored → -log(1-F_marg(c))")
    print("  KM-ASSE method: km_plotting_positions(failures, runouts) → KM plotting probs")
    print("  Both: same 40-pt GH quadrature, no change to marginal CDF computation.")
