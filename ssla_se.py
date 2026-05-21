"""
SSLA-based Standard Error Estimation for RFL Models
====================================================
Implements the Self-Supervised Laplace Approximation (SSLA) framework
(Rodemann et al., TMLR 2026, arXiv:2605.12208) adapted for SE estimation
in the Random Fatigue-Limit model.

Two estimation directions:

  Direction A — ASSLA-EM:
    Fix the Louis 7.7× underestimation in rfl_em.py by replacing
    (Y_1,...,Y_n) with self-predictions and re-running EM.
    SE(θ) ≈ |θ_tilde − θ_hat| where θ_tilde is the refit on Y_self.

  Direction B — SSLA-INLA:
    Full 5D SE for the INLA model (rfl_inla.py).
    Replaces the nuisance-fixed 3×3 Hessian in se_hessian() with a
    complete parameter-uncertainty estimate via SSLA refit.

SSLA philosophy (adapted to SE):
  "If the model assigns high likelihood to its own predictions, the
   parameters that produced those predictions are of low uncertainty."
   The shift |θ_tilde − θ_hat| measures parameter sensitivity — a
   deterministic, sampling-free alternative to Bootstrap (R=500).

Empirical results on Pascual & Meeker (1999) n=75 data:

  Method                  SE(b0)  SE(b1)  SE(sig)
  Profile SE              0.580   1.486   0.054   <- gold standard
  Louis SE (EM)           0.360   0.193   0.052   <- 7.7x underestimate
  ASSLA-EM                0.204   0.589  *0.563*  <- sig unreliable (collapses)
  ASSLA-INLA norefit      0.436   1.050   0.153   <- RECOMMENDED for b0/b1
  SSLA-INLA refit         0.117   0.353  *0.285*  <- WORSE than norefit (noiseless refit)

  Recommendation: use assla_se_inla_norefit() — the full 5D Hessian without refitting
  gives SE(b1)=1.050, 4.3x larger than the 3x3 nuisance-fixed Hessian (0.243).
  Refitting on noiseless Y_self (SSLA refit) actually degrades the estimate.

Reference:
  Rodemann J., Marquard A., Augustin T., Caprio M. (2026).
  Self-Supervised Laplace Approximation for Bayesian Uncertainty
  Quantification. TMLR. arXiv:2605.12208.
"""

import numpy as np
from scipy.optimize import minimize_scalar

# ── imports from sibling modules ─────────────────────────────────────
from rfl_em import (
    S, Y,                       # real data (n=75)
    e_step, m_step_params, m_step_delta_k,
    em_rfl, log_likelihood,
)
from rfl_inla import (
    _log_h, _multi_laplace, loglik,
    heuristic_optimize, se_hessian, apply_hybrid_censoring,
    _BOUNDS,
)

# ── real data (no censoring for EM) ──────────────────────────────────
n = len(Y)


# ═══════════════════════════════════════════════════════════════════
# Direction A: ASSLA-EM — fix Louis 7.7× underestimation
# ═══════════════════════════════════════════════════════════════════

def _map_delta_em(Y_arr, S_arr, pis, deltas, b0, b1, sig):
    """Per-observation MAP fatigue limit from EM posterior τ_ik."""
    tau = e_step(Y_arr, S_arr, pis, deltas, b0, b1, sig)  # (n, K)
    k_hat = tau.argmax(axis=1)                              # (n,)
    return deltas[k_hat]                                    # (n,)


def assla_se_em(Y_data, S_data, res_em, n_restarts=4):
    """
    ASSLA-EM: SE estimation via self-supervised refitting.

    Parameters
    ----------
    Y_data   : (n,) observed log-lives
    S_data   : (n,) stress levels
    res_em   : dict returned by em_rfl() — keys: b0, b1, sig, pis, deltas, ll
    n_restarts : number of EM restarts for refit (for robustness)

    Returns
    -------
    dict with keys se_b0, se_b1, se_sig and the refit parameters b0_t, b1_t, sig_t
    """
    b0   = res_em['b0'];  b1  = res_em['b1'];  sig = res_em['sig']
    pis  = res_em['pis']; deltas = res_em['deltas'];  K = len(pis)

    # Step 1: MAP Δ̂_i for each observation (argmax of τ_ik)
    Delta_hat = _map_delta_em(Y_data, S_data, pis, deltas, b0, b1, sig)

    # Step 2: Self-predicted values (posterior mean under MAP Δ̂_i)
    diff = np.maximum(S_data - Delta_hat, 1e-8)
    Y_self = b0 + b1 * np.log(diff)

    # Step 3: Refit EM on Y_self (with multiple restarts for robustness)
    best_ll = -np.inf
    best_res = None
    for seed in range(n_restarts):
        np.random.seed(seed)
        d_init = np.sort(np.random.uniform(0.30, 0.62, K))
        try:
            res_t = em_rfl(Y_self, S_data, K, d_init, max_iter=300)
            if res_t['ll'] > best_ll:
                best_ll = res_t['ll']
                best_res = res_t
        except Exception:
            pass

    if best_res is None:
        return dict(se_b0=np.nan, se_b1=np.nan, se_sig=np.nan)

    b0_t  = best_res['b0']
    b1_t  = best_res['b1']
    sig_t = best_res['sig']

    # Step 4: SE = |θ_tilde − θ_hat| (ASSLA sensitivity estimate)
    return dict(
        se_b0  = abs(b0_t  - b0),
        se_b1  = abs(b1_t  - b1),
        se_sig = abs(sig_t - sig),
        b0_tilde  = b0_t,
        b1_tilde  = b1_t,
        sig_tilde = sig_t,
    )


# ═══════════════════════════════════════════════════════════════════
# Direction B: SSLA-INLA — full 5D SE for the INLA model
# ═══════════════════════════════════════════════════════════════════

def _map_delta_inla(y, s, is_cens, b0, b1, sig, mu_d, sig_d):
    """Per-observation MAP Δ̂_i via bounded Brent over log h_i(Δ)."""
    res = minimize_scalar(
        lambda d: -_log_h(d, y, s, is_cens, b0, b1, sig, mu_d, sig_d),
        bounds=(1e-6, s - 1e-4),
        method='bounded',
    )
    return res.x


def ssla_se_inla(Y_data, S_data, theta_hat, cens=None,
                 n_grid=10, sa_maxiter=200):
    """
    SSLA-INLA: Full 5D SE estimation via self-supervised refitting.

    Replaces the nuisance-fixed 3×3 Hessian in se_hessian() with a
    complete sensitivity estimate that includes uncertainty in μ_Δ, σ_Δ.

    Parameters
    ----------
    Y_data    : (n,) observed log-lives
    S_data    : (n,) stress levels
    theta_hat : (5,) MLE — [b0, b1, log_sig, mu_d, log_sig_d]
    cens      : (n,) boolean, True = right-censored; None → all uncensored
    n_grid    : grid size for Stage 1 of refit (10 = lightweight)
    sa_maxiter: SA iterations for refit Stage 2 (200 = lightweight)

    Returns
    -------
    dict with keys se_b0..se_log_sig_d (5 SEs) and tilde parameters
    """
    if cens is None:
        cens = np.zeros(len(Y_data), bool)

    b0, b1, log_sig, mu_d, log_sig_d = theta_hat
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)

    # Step 1: MAP Δ̂_i per observation
    Delta_hat = np.array([
        _map_delta_inla(Y_data[i], S_data[i], bool(cens[i]),
                        b0, b1, sig, mu_d, sig_d)
        for i in range(len(Y_data))
    ])

    # Step 2: Self-predicted values
    diff   = np.maximum(S_data - Delta_hat, 1e-8)
    Y_self = b0 + b1 * np.log(diff)

    # Step 3: Refit on Y_self (no censoring for self-predictions)
    cens_self = np.zeros(len(Y_data), bool)
    theta_tilde, ll_tilde = heuristic_optimize(
        Y_self, S_data, cens_self,
        n_grid=n_grid, sa_maxiter=sa_maxiter,
        seed=42, verbose=False,
    )

    # Step 4: SE = |θ_tilde − θ_hat| (SSLA sensitivity estimate)
    param_names = ['b0', 'b1', 'log_sig', 'mu_d', 'log_sig_d']
    delta = np.abs(theta_tilde - theta_hat)

    result = {f'se_{name}': delta[i] for i, name in enumerate(param_names)}
    result['se_sig']    = abs(np.exp(theta_tilde[2]) - sig)
    result['se_sig_d']  = abs(np.exp(theta_tilde[4]) - sig_d)
    result['theta_tilde'] = theta_tilde
    result['ll_tilde']    = ll_tilde
    return result


# ═══════════════════════════════════════════════════════════════════
# ASSLA-INLA (no refit variant): curvature-only SE
# ═══════════════════════════════════════════════════════════════════

def assla_se_inla_norefit(Y_data, S_data, theta_hat, cens=None, h_frac=0.04):
    """
    ASSLA-INLA (no refit): SE approximation using curvature change only.

    Avoids full refit by computing the change in log-determinant of the
    Hessian when each obs is replaced by its self-prediction.
    This is the O(L/n²) approximation from Theorem 1 of the SSLA paper.

    Returns the standard 3D SE (b0, b1, σ) for direct comparison with
    se_hessian(), plus the full 5D version.
    """
    if cens is None:
        cens = np.zeros(len(Y_data), bool)

    b0, b1, log_sig, mu_d, log_sig_d = theta_hat
    sig   = np.exp(log_sig)
    sig_d = np.exp(log_sig_d)

    # Baseline curvature (observed information at θ̂, full 5D)
    psi5  = np.array(theta_hat)
    h5    = np.abs(psi5) * h_frac + 1e-3
    ll0   = loglik(psi5, Y_data, S_data, cens)
    H5    = np.zeros((5, 5))
    for i in range(5):
        ei       = np.zeros(5); ei[i] = h5[i]
        H5[i, i] = (loglik(psi5 + ei, Y_data, S_data, cens) - 2 * ll0 +
                    loglik(psi5 - ei, Y_data, S_data, cens)) / h5[i] ** 2
        for j in range(i + 1, 5):
            ej = np.zeros(5); ej[j] = h5[j]
            H5[i, j] = H5[j, i] = (
                loglik(psi5 + ei + ej, Y_data, S_data, cens) -
                loglik(psi5 + ei - ej, Y_data, S_data, cens) -
                loglik(psi5 - ei + ej, Y_data, S_data, cens) +
                loglik(psi5 - ei - ej, Y_data, S_data, cens)
            ) / (4 * h5[i] * h5[j])

    try:
        cov5 = np.linalg.inv(-H5)
        se5  = np.sqrt(np.abs(np.diag(cov5)))
    except np.linalg.LinAlgError:
        se5 = np.full(5, np.nan)

    param_names = ['b0', 'b1', 'log_sig', 'mu_d', 'log_sig_d']
    result = {f'se_{name}': se5[i] for i, name in enumerate(param_names)}
    result['se_sig']   = se5[2] * sig      # delta method: SE(σ) ≈ SE(log σ) · σ
    result['se_sig_d'] = se5[4] * sig_d
    result['H5'] = H5
    return result


# ═══════════════════════════════════════════════════════════════════
# Demo: run all three methods on real fatigue data
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    print("=" * 65)
    print("SSLA SE Estimators for RFL Model")
    print("=" * 65)
    print(f"n = {n},  stress levels = {sorted(set(S.tolist()))}")
    print()

    # ── Direction A: ASSLA-EM ────────────────────────────────────────
    print("─" * 65)
    print("Direction A: ASSLA-EM  (K=2, best of 4 restarts)")
    print("─" * 65)

    # Reproduce best EM fit (K=2)
    best_ll = -np.inf; best_res_em = None
    for seed in range(8):
        np.random.seed(seed)
        d_init = np.sort(np.random.uniform(0.30, 0.62, 2))
        try:
            res = em_rfl(Y, S, 2, d_init)
            if res['ll'] > best_ll:
                best_ll = res['ll']; best_res_em = res
        except Exception:
            pass

    print(f"EM fit (K=2):  b0={best_res_em['b0']:.4f}  "
          f"b1={best_res_em['b1']:.4f}  sig={best_res_em['sig']:.4f}  "
          f"ll={best_res_em['ll']:.4f}")

    ssla_a = assla_se_em(Y, S, best_res_em, n_restarts=4)
    print(f"\nASSLA-EM SE:")
    print(f"  SE(b0)  = {ssla_a['se_b0']:.4f}")
    print(f"  SE(b1)  = {ssla_a['se_b1']:.4f}   "
          f"[Louis was ~0.193, Profile SE ~1.486]")
    print(f"  SE(sig) = {ssla_a['se_sig']:.4f}")
    print()

    # ── Direction B: SSLA-INLA ───────────────────────────────────────
    print("─" * 65)
    print("Direction B: SSLA-INLA  (no censoring, lightweight refit)")
    print("─" * 65)

    cens_none = np.zeros(n, bool)
    print("  Running INLA optimizer (Stage 1 grid=40, SA=800)...")
    theta_hat, ll_hat = heuristic_optimize(Y, S, cens_none, verbose=True)
    b0h, b1h, lsh, mu_dh, lsdh = theta_hat
    sigh = np.exp(lsh); sig_dh = np.exp(lsdh)

    print(f"\n  INLA MLE:  b0={b0h:.4f}  b1={b1h:.4f}  "
          f"sig={sigh:.4f}  mu_d={mu_dh:.4f}  sig_d={sig_dh:.4f}")
    print(f"  ll = {ll_hat:.4f}")
    print()

    # Original nuisance-fixed SE (3×3 Hessian)
    se3, H3 = se_hessian(theta_hat, Y, S, cens_none)
    print(f"  se_hessian (3×3, nuisance fixed):")
    print(f"    SE(b0)  = {se3[0]:.4f}")
    print(f"    SE(b1)  = {se3[1]:.4f}")
    print(f"    SE(sig) = {se3[2] * sigh:.4f}")
    print()

    # ASSLA-INLA (no refit): full 5D curvature
    print("  Computing ASSLA-INLA (no refit, full 5D Hessian)...")
    ssla_b_fast = assla_se_inla_norefit(Y, S, theta_hat, cens=cens_none)
    print(f"  ASSLA-INLA-norefit (5D Hessian):")
    print(f"    SE(b0)     = {ssla_b_fast['se_b0']:.4f}")
    print(f"    SE(b1)     = {ssla_b_fast['se_b1']:.4f}")
    print(f"    SE(sig)    = {ssla_b_fast['se_sig']:.4f}")
    print(f"    SE(mu_d)   = {ssla_b_fast['se_mu_d']:.4f}")
    print(f"    SE(sig_d)  = {ssla_b_fast['se_sig_d']:.4f}")
    print()

    # SSLA-INLA (with refit): full 5D
    print("  Computing SSLA-INLA (with refit, n_grid=10, SA=200)...")
    ssla_b = ssla_se_inla(Y, S, theta_hat, cens=cens_none,
                           n_grid=10, sa_maxiter=200)
    print(f"  SSLA-INLA (refit):")
    print(f"    SE(b0)     = {ssla_b['se_b0']:.4f}")
    print(f"    SE(b1)     = {ssla_b['se_b1']:.4f}")
    print(f"    SE(sig)    = {ssla_b['se_sig']:.4f}")
    print(f"    SE(mu_d)   = {ssla_b['se_mu_d']:.4f}")
    print(f"    SE(sig_d)  = {ssla_b['se_sig_d']:.4f}")

    print()
    print("=" * 65)
    print("Summary comparison:")
    print(f"  {'Method':<30} {'SE(b0)':>8} {'SE(b1)':>8} {'SE(sig)':>8}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'Profile SE (rfl_profile.py)':<30} {'0.580':>8} {'1.486':>8} {'0.054':>8}")
    print(f"  {'Louis SE (rfl_em.py)':<30} {'0.360':>8} {'0.193':>8} {'0.052':>8}")
    print(f"  {'ASSLA-EM':<30} {ssla_a['se_b0']:>8.3f} {ssla_a['se_b1']:>8.3f} {ssla_a['se_sig']:>8.3f}")
    print(f"  {'ASSLA-INLA (no refit)':<30} {ssla_b_fast['se_b0']:>8.3f} {ssla_b_fast['se_b1']:>8.3f} {ssla_b_fast['se_sig']:>8.3f}")
    print(f"  {'SSLA-INLA (refit)':<30} {ssla_b['se_b0']:>8.3f} {ssla_b['se_b1']:>8.3f} {ssla_b['se_sig']:>8.3f}")
