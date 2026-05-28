"""
rfl_asse_zscore.py — Roy's z-score ASSE vs rank-based ASSE(n=75)
================================================================
No fitting. Uses hardcoded known parameter estimates.

Z-score approach:
  z_ij     = (lnY_ij - mean_j) / std_j   within each stress group
  delta_ij = exp(mu_d + sig_d * z_ij)    LogNormal quantile at Phi(z_ij)
  yhat_ij  = b0 + b1 * ln(s_j - delta_ij) [- euler_shift for SEV]
  ASSE_z   = sum |lnY_ij - yhat_ij|
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfl_compare_all import (
    S_all, Y_all,
    _GHX, _GHW, _SQRT2PI, EULER_GAMMA,
    compute_E, marg_quantile,
    inla_gh_marg_cdf, sev_cdf, normal_cdf,
    burr_inla_marg_cdf,
)

# ── Hardcoded fitted parameters (from prior runs) ────────────────────────────
# theta = [b0, b1, log_sig, mu_d, log_sig_d]
# SEV+INLA   : rfl_burr_inla.py line 183 comment
# SEV+MCEM   : rfl_sev_inla.py line 175 (MCEM result used as warm-start)
# Burr+INLA  : approx from SEV+INLA base + sig/a from README

THETA_NORM = np.array([-9.37,  -8.348, np.log(0.295), -0.635, np.log(0.033)])
THETA_SEV  = np.array([-9.316, -8.534, np.log(0.190), -0.645, np.log(0.036)])
THETA_MCEM = np.array([-8.480, -6.592, np.log(0.268), -0.540, np.log(0.029)])
THETA_BURR = np.array([-9.316, -8.534, np.log(0.160), np.log(1.62), -0.645, np.log(0.036)])

# Burr+EM-GMM K=1 Mode A  (from rfl_burr_em.py, sig>=0.15 a>=1 constraints)
BURR_EM_B0    = -9.382742
BURR_EM_B1    = -8.513044
BURR_EM_SIG   =  0.161091   # SEV scale
BURR_EM_MU1   =  0.52558521  # Normal GMM mean (K=1)
BURR_EM_SIG1  =  0.01846682  # Normal GMM std  (K=1)

# ── Z-score ASSE ─────────────────────────────────────────────────────────────
def compute_E_zscore(Y_log, S_arr, b0, b1, mu_d, sig_d, euler_shift=0.0):
    E_total = 0.0
    skipped = 0
    for s in np.unique(S_arr):
        idx     = S_arr == s
        y_group = np.sort(Y_log[idx])
        n_j     = len(y_group)
        mean_j  = y_group.mean()
        std_j   = y_group.std(ddof=1)
        for y_ij in y_group:
            z_ij     = (y_ij - mean_j) / std_j
            delta_ij = np.exp(mu_d + sig_d * z_ij)
            if delta_ij >= s - 1e-8 or delta_ij <= 0:
                skipped += 1
                continue
            yhat_ij = b0 + b1 * np.log(s - delta_ij) - euler_shift
            E_total += abs(y_ij - yhat_ij)
    if skipped:
        print(f"  [zscore] skipped {skipped} obs (delta >= s)")
    return E_total

# ── Z-score ASSE (Normal GMM prior) ──────────────────────────────────────────
def compute_E_zscore_normal(Y_log, S_arr, b0, b1, mu_1, sig_1, euler_shift=0.0):
    """Burr+EM-GMM K=1: delta_ij = mu_1 + sig_1 * z_ij  (Normal quantile)"""
    E_total = 0.0
    skipped = 0
    for s in np.unique(S_arr):
        idx     = S_arr == s
        y_group = np.sort(Y_log[idx])
        mean_j  = y_group.mean()
        std_j   = y_group.std(ddof=1)
        for y_ij in y_group:
            z_ij     = (y_ij - mean_j) / std_j
            delta_ij = mu_1 + sig_1 * z_ij       # Normal quantile
            if delta_ij >= s - 1e-8 or delta_ij <= 0:
                skipped += 1
                continue
            yhat_ij = b0 + b1 * np.log(s - delta_ij) - euler_shift
            E_total += abs(y_ij - yhat_ij)
    if skipped:
        print(f"  [zscore_normal] skipped {skipped} obs (delta >= s or <=0)")
    return E_total

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Using hardcoded parameters (no fitting)")
    for name, th in [("Normal+INLA", THETA_NORM),
                     ("SEV+INLA",    THETA_SEV),
                     ("SEV+MCEM",    THETA_MCEM),
                     ("Burr+INLA",   THETA_BURR)]:
        n = len(th)
        b0, b1 = th[0], th[1]
        sig = np.exp(th[2])
        mu_d  = th[n-2]
        sig_d = np.exp(th[n-1])
        print(f"  {name}: b0={b0:.4f}  b1={b1:.4f}  sig={sig:.4f}  "
              f"mu_d={mu_d:.4f}  sig_d={sig_d:.4f}")

    # ── Rank-based ASSE(n=75) ─────────────────────────────────────────────────
    print("\n--- Rank-based ASSE(n=75) [(i-0.5)/n → marginal CDF inversion] ---")
    norm_qtl = lambda p, s: marg_quantile(
        p, s, lambda y, ss: inla_gh_marg_cdf(y, ss, THETA_NORM, normal_cdf))
    sev_qtl  = lambda p, s: marg_quantile(
        p, s, lambda y, ss: inla_gh_marg_cdf(y, ss, THETA_SEV,  sev_cdf))
    mcem_qtl = lambda p, s: marg_quantile(
        p, s, lambda y, ss: inla_gh_marg_cdf(y, ss, THETA_MCEM, sev_cdf))
    burr_qtl = lambda p, s: marg_quantile(
        p, s, lambda y, ss: burr_inla_marg_cdf(y, ss, THETA_BURR))

    E_norm_rank = compute_E(Y_all, S_all, norm_qtl)
    E_sev_rank  = compute_E(Y_all, S_all, sev_qtl)
    E_mcem_rank = compute_E(Y_all, S_all, mcem_qtl)
    E_burr_rank = compute_E(Y_all, S_all, burr_qtl)
    print(f"  Normal+INLA : {E_norm_rank:.3f}  (expect ~12.85)")
    print(f"  SEV+INLA    : {E_sev_rank:.3f}  (expect ~13.02)")
    print(f"  SEV+MCEM    : {E_mcem_rank:.3f}")
    print(f"  Burr+INLA   : {E_burr_rank:.3f}  (expect ~12.78)")

    # ── Z-score ASSE ─────────────────────────────────────────────────────────
    print("\n--- Z-score ASSE [z=(lnY-mean)/std → delta=exp(mu_d+sig_d*z) → yhat] ---")
    E_norm_z = compute_E_zscore(Y_all, S_all,
                                 THETA_NORM[0], THETA_NORM[1],
                                 THETA_NORM[3], np.exp(THETA_NORM[4]),
                                 euler_shift=0.0)
    E_sev_z  = compute_E_zscore(Y_all, S_all,
                                 THETA_SEV[0],  THETA_SEV[1],
                                 THETA_SEV[3],  np.exp(THETA_SEV[4]),
                                 euler_shift=np.exp(THETA_SEV[2]) * EULER_GAMMA)
    E_mcem_z = compute_E_zscore(Y_all, S_all,
                                 THETA_MCEM[0], THETA_MCEM[1],
                                 THETA_MCEM[3], np.exp(THETA_MCEM[4]),
                                 euler_shift=np.exp(THETA_MCEM[2]) * EULER_GAMMA)
    E_burr_z = compute_E_zscore(Y_all, S_all,
                                 THETA_BURR[0], THETA_BURR[1],
                                 THETA_BURR[4], np.exp(THETA_BURR[5]),
                                 euler_shift=0.0)
    E_burrem_z = compute_E_zscore_normal(
                                 Y_all, S_all,
                                 BURR_EM_B0, BURR_EM_B1,
                                 BURR_EM_MU1, BURR_EM_SIG1,
                                 euler_shift=BURR_EM_SIG * EULER_GAMMA)
    print(f"  Normal+INLA  : {E_norm_z:.3f}")
    print(f"  SEV+INLA     : {E_sev_z:.3f}")
    print(f"  SEV+MCEM     : {E_mcem_z:.3f}")
    print(f"  Burr+INLA    : {E_burr_z:.3f}")
    print(f"  Burr+EM-GMM  : {E_burrem_z:.3f}  [Normal prior, K=1 Mode A]")

    # ── Rank-based for Burr+EM-GMM (uses per-obs marginal, no closed form here)
    print("\n  Note: Burr+EM-GMM rank-based ASSE not computed (no marg_quantile)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Model':<22} {'rank-based':>12} {'z-score':>10} {'diff':>8}")
    print(f"  {'-'*56}")
    for name, e_r, e_z in [
        ("Normal+INLA",  E_norm_rank, E_norm_z),
        ("SEV+INLA",     E_sev_rank,  E_sev_z),
        ("SEV+MCEM",     E_mcem_rank, E_mcem_z),
        ("Burr+INLA",    E_burr_rank, E_burr_z),
        ("Burr+EM-GMM",  None,        E_burrem_z),
    ]:
        if e_r is not None:
            print(f"  {name:<22} {e_r:>12.3f} {e_z:>10.3f} {e_z - e_r:>+8.3f}")
        else:
            print(f"  {name:<22} {'—':>12} {e_z:>10.3f} {'—':>8}")
    print(f"\n  Thesis (Chiu 2005) ASSE = 10.80")
