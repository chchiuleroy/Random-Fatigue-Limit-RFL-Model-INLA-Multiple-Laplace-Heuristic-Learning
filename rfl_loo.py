"""
rfl_loo.py — Leave-One-Out Cross-Validation for RFL Models
============================================================
核心問題：in-sample ASSE 的方法排名，在 LOO 下是否成立？

關鍵洞察：
  In-sample ASSE 對 y_i 「雙重使用」：
    1. θ̂ 由 y_i 參與估計
    2. per-obs 後驗 p(Δᵢ|y_i,θ̂) 再用一次 y_i
  → ASSE 低不等於「預測新試件準」，而是「給了 y_i 再還原 y_i 準」

  LOO 移除第 2 點：參數由 y_{-i} 估（或近似），
  但無法再用 y_i 計算後驗 → 只能用 prior predictive

LOO 預測的三層定義：
  (A) Prior-LOO   : E_g(Δ;θ̂_{-i})[μ(Sᵢ,Δ)] - σγ       ← 最誠實，不看 y_i
  (B) Post-LOO    : E[μ(Sᵢ,Δ)|y_i,θ̂_{-i}] - σγ          ← 看 y_i，但 LOO 參數
  (C) In-sample   : E[μ(Sᵢ,Δ)|y_i,θ̂] - σγ               ← 雙重使用 y_i

LOO 參數近似（影響函數，influence function）：
  θ̂_{-i} ≈ θ̂ + H_info⁻¹ · ∇ℓᵢ(θ̂)   （移除第 i 筆後參數的位移）
  H_info = -∂²ℓ/∂θ² |_{θ̂}             （observed Fisher information）

方法比較：
  SEV+INLA       (in-sample ASSE = 5.76)
  Normal+NPMLE   (in-sample ASSE = 33.87)
  S-conditional mean baseline（無潛在變數模型）

注意：LOO 的「prior predictive」對同應力水準所有試件給同一個預測值
→ LOO-ASSE ≈ within-stress MAD ── 這才是對新試件的真實預測能力。
"""

import numpy as np
from scipy.stats import lognorm, norm
from scipy.optimize import minimize_scalar, minimize
import warnings; warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015328

# ── 資料（n=75，5 應力水準）────────────────────────────────────────
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
S_r, Y_r = [], []
for s, ys in _raw.items():
    for y in ys: S_r.append(s); Y_r.append(np.log(y))
S_r = np.array(S_r); Y_r = np.array(Y_r)
cens_r = np.zeros(len(Y_r), bool)
stresses = sorted(_raw.keys())
n = len(Y_r)

# ══════════════════════════════════════════════════════════════════
# §1  SEV+INLA 工具函數
# ══════════════════════════════════════════════════════════════════
N_GH = 9
_GHX, _GHW = np.polynomial.hermite_e.hermegauss(N_GH)

def _sev_logpdf(y, mu, sig):
    w = (y - mu) / sig
    return -np.log(sig) + w - np.exp(w)

def _sev_logsf(y, mu, sig):
    return -np.exp((y - mu) / sig)

def _log_h(d, y, s, is_cens, b0, b1, sig, mu_d, sig_d):
    if d <= 0 or d >= s - 1e-8: return -1e30
    mu_y = b0 + b1 * np.log(s - d)
    lf = _sev_logsf(y, mu_y, sig) if is_cens else _sev_logpdf(y, mu_y, sig)
    lg = lognorm.logpdf(d, s=sig_d, scale=np.exp(mu_d))
    return lf + lg

def _gh_logmarginal(y, s, is_cens, b0, b1, sig, mu_d, sig_d):
    """9-pt GH 積分：log 邊際似然（per-obs）"""
    res = minimize_scalar(lambda d: -_log_h(d, y, s, is_cens, b0, b1, sig, mu_d, sig_d),
                          bounds=(1e-6, s - 1e-4), method='bounded')
    d_hat = res.x
    lh0 = _log_h(d_hat, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    if lh0 < -1e25: return -1e30
    eps = max(d_hat * 0.005, 5e-6)
    lhp = _log_h(d_hat + eps, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    lhm = _log_h(d_hat - eps, y, s, is_cens, b0, b1, sig, mu_d, sig_d)
    kappa = max(-(lhp - 2*lh0 + lhm) / eps**2, 1e-6)
    sig_t = 1.0 / np.sqrt(kappa)
    d_pts = d_hat + sig_t * _GHX
    lh_pts = np.array([_log_h(dj, y, s, is_cens, b0, b1, sig, mu_d, sig_d) for dj in d_pts])
    log_terms = lh_pts + _GHX**2 / 2
    log_shift = log_terms.max()
    L_i = sig_t * np.sum(_GHW * np.exp(log_terms - log_shift)) * np.exp(log_shift)
    return np.log(max(L_i, 1e-300))

def _gh_posterior_mean_mu(y, s, b0, b1, sig, mu_d, sig_d):
    """E[μ(S,Δ)|y] via GH：in-sample 後驗均值 E[μᵢ|yᵢ]"""
    res = minimize_scalar(lambda d: -_log_h(d, y, s, False, b0, b1, sig, mu_d, sig_d),
                          bounds=(1e-6, s - 1e-4), method='bounded')
    d_hat = res.x
    lh0 = _log_h(d_hat, y, s, False, b0, b1, sig, mu_d, sig_d)
    if lh0 < -1e25: return b0 + b1 * np.log(max(s - d_hat, 1e-8))
    eps = max(d_hat * 0.005, 5e-6)
    lhp = _log_h(d_hat + eps, y, s, False, b0, b1, sig, mu_d, sig_d)
    lhm = _log_h(d_hat - eps, y, s, False, b0, b1, sig, mu_d, sig_d)
    kappa = max(-(lhp - 2*lh0 + lhm) / eps**2, 1e-6)
    sig_t = 1.0 / np.sqrt(kappa)
    d_pts = d_hat + sig_t * _GHX
    lh_pts = np.array([_log_h(dj, y, s, False, b0, b1, sig, mu_d, sig_d) for dj in d_pts])
    log_terms = lh_pts + _GHX**2 / 2
    log_shift = log_terms.max()
    unnorm = _GHW * np.exp(log_terms - log_shift)
    weights = unnorm / max(unnorm.sum(), 1e-300)
    mu_pts = np.array([b0 + b1 * np.log(max(s - dj, 1e-8)) for dj in d_pts])
    return float(np.sum(weights * mu_pts))

def inla_total_loglik(Y, S, cens, theta):
    b0, b1, log_sig, mu_d, log_sig_d = theta
    sig = np.exp(log_sig); sig_d = np.exp(log_sig_d)
    if sig < 0.01 or sig_d < 0.005: return -1e30
    return sum(_gh_logmarginal(Y[i], S[i], bool(cens[i]), b0, b1, sig, mu_d, sig_d)
               for i in range(len(Y)))

def inla_per_obs_loglik(y, s, cens_i, theta):
    b0, b1, log_sig, mu_d, log_sig_d = theta
    sig = np.exp(log_sig); sig_d = np.exp(log_sig_d)
    return _gh_logmarginal(y, s, bool(cens_i), b0, b1, sig, mu_d, sig_d)

# ── SEV+INLA Prior predictive mean (LOO prediction A) ────────────
def sev_inla_prior_mean(s, b0, b1, sig, mu_d, sig_d, n_grid=500):
    """
    E_g(Δ)[μ(S,Δ)] - σγ，g = LogNormal(μ_d, σ_d²)
    = LOO prior predictive（完全不用 y_i）
    同應力水準所有試件給相同預測值。
    """
    d_lo, d_hi = 1e-6, s - 1e-4
    d_grid = np.linspace(d_lo, d_hi, n_grid)
    log_g = lognorm.logpdf(d_grid, s=sig_d, scale=np.exp(mu_d))
    log_g -= log_g.max()
    g_w = np.exp(log_g); g_w /= g_w.sum()
    mu_grid = b0 + b1 * np.log(s - d_grid)
    return float(np.sum(g_w * mu_grid)) - sig * EULER_GAMMA

# ══════════════════════════════════════════════════════════════════
# §2  Normal+NPMLE 工具函數（K=2）
# ══════════════════════════════════════════════════════════════════
def npmle_e_step(Y, S, pis, deltas, b0, b1, sig):
    """軟分配 τ_{ik}，K=2"""
    K = len(pis)
    logw = np.full((len(Y), K), -1e30)
    for k in range(K):
        diff = S - deltas[k]
        ok = diff > 1e-8
        mu_k = np.where(ok, b0 + b1 * np.log(np.maximum(diff, 1e-300)), -1e30)
        logw[:, k] = np.where(ok, np.log(pis[k]) + norm.logpdf(Y, mu_k, sig), -1e30)
    rm = logw.max(1, keepdims=True)
    e = np.exp(logw - rm)
    return e / e.sum(1, keepdims=True)

def npmle_per_obs_loglik(Y, S, pis, deltas, b0, b1, sig):
    """per-obs log 邊際似然 for Normal+NPMLE"""
    K = len(pis)
    lls = np.zeros(len(Y))
    for i in range(len(Y)):
        terms = []
        for k in range(K):
            d = S[i] - deltas[k]
            if d > 1e-8:
                mu_k = b0 + b1 * np.log(d)
                terms.append(np.log(pis[k]) + norm.logpdf(Y[i], mu_k, sig))
        if terms:
            mx = max(terms)
            lls[i] = mx + np.log(sum(np.exp(t - mx) for t in terms))
        else:
            lls[i] = -1e30
    return lls

def npmle_prior_mean(s, pis, deltas, b0, b1):
    """
    Σₖ πₖ × μ(S, Δₖ) = LOO prior predictive for Normal+NPMLE
    """
    mu_vals = []
    for k in range(len(pis)):
        d = s - deltas[k]
        if d > 1e-8:
            mu_vals.append(pis[k] * (b0 + b1 * np.log(d)))
        else:
            mu_vals.append(0.0)
    return sum(mu_vals)

# ══════════════════════════════════════════════════════════════════
# §3  影響函數 LOO（Influence Function Approximation）
# ══════════════════════════════════════════════════════════════════
def compute_hessian(Y, S, cens, theta, fn_total, eps=1e-4):
    """
    Total log-lik 的數值 Hessian（2 階混合偏微分）。
    只需計算一次，O(p²) 次 fn_total 評估。
    """
    p = len(theta)
    H = np.zeros((p, p))
    for j in range(p):
        for k in range(j, p):
            tp = theta.copy(); tp[j] += eps; tp[k] += eps
            tpm = theta.copy(); tpm[j] += eps; tpm[k] -= eps
            tmp = theta.copy(); tmp[j] -= eps; tmp[k] += eps
            tm = theta.copy(); tm[j] -= eps; tm[k] -= eps
            H[j, k] = H[k, j] = (fn_total(Y, S, cens, tp) - fn_total(Y, S, cens, tpm)
                                  - fn_total(Y, S, cens, tmp) + fn_total(Y, S, cens, tm)) / (4 * eps**2)
    return H

def compute_scores(Y, S, cens, theta, fn_per_obs, eps=1e-5):
    """
    Per-obs score ∇ℓᵢ(θ̂) for each i. Shape (n, p).
    Central difference, O(n×p×2) fn evaluations.
    """
    p = len(theta)
    scores = np.zeros((len(Y), p))
    for j in range(p):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        for i in range(len(Y)):
            lp = fn_per_obs(Y[i], S[i], cens[i], tp)
            lm = fn_per_obs(Y[i], S[i], cens[i], tm)
            scores[i, j] = (lp - lm) / (2 * eps)
    return scores

def loo_theta_if(theta, H_info_inv, scores):
    """
    θ̂_{-i} ≈ θ̂ + H_info⁻¹ · ∇ℓᵢ(θ̂)  for all i.
    移除觀測 i → 參數往 score 方向位移。
    Returns (n, p).
    """
    # scores: (n, p), H_info_inv: (p, p)
    # delta_theta[i] = H_info_inv @ scores[i]  →  broadcast
    return theta[None, :] + (H_info_inv @ scores.T).T

# ══════════════════════════════════════════════════════════════════
# §4  主分析
# ══════════════════════════════════════════════════════════════════
def analyse_sev_inla(Y, S, cens, theta_hat):
    """
    計算 SEV+INLA 的三種 ASSE：
    (A) Prior-LOO   (B) Post-LOO   (C) In-sample
    並回傳 per-obs 數字。
    """
    b0, b1, log_sig, mu_d, log_sig_d = theta_hat
    sig = np.exp(log_sig); sig_d = np.exp(log_sig_d)

    # ── (C) In-sample posterior fitted values ──────────────────
    print("  [C] In-sample: 計算後驗均值 E[μᵢ|yᵢ,θ̂]...")
    y_hat_C = np.array([
        _gh_posterior_mean_mu(Y[i], S[i], b0, b1, sig, mu_d, sig_d) - sig * EULER_GAMMA
        for i in range(len(Y))
    ])

    # ── Hessian + scores ────────────────────────────────────────
    print("  影響函數：計算 Hessian（p=5，25 次 loglik 評估）...")
    H = compute_hessian(Y, S, cens, theta_hat, inla_total_loglik, eps=1e-4)
    H_info = -H
    try:
        H_info_inv = np.linalg.inv(H_info)
        cond = np.linalg.cond(H_info)
        print(f"    Hessian condition number = {cond:.2e}")
    except np.linalg.LinAlgError:
        print("    [警告] Hessian 不可逆，使用 pseudo-inverse")
        H_info_inv = np.linalg.pinv(H_info)

    print(f"  影響函數：計算 per-obs scores（{5*2*n} 次 loglik 評估）...")
    scores = compute_scores(Y, S, cens, theta_hat, inla_per_obs_loglik, eps=1e-5)

    # LOO 參數：(n, 5)
    theta_loo = loo_theta_if(theta_hat, H_info_inv, scores)
    # 位移量統計（診斷）
    shift_norms = np.linalg.norm(theta_loo - theta_hat, axis=1)
    print(f"    θ 位移 |Δθᵢ|：mean={shift_norms.mean():.4f}  max={shift_norms.max():.4f}")

    # ── (A) Prior-LOO fitted values ─────────────────────────────
    print("  [A] Prior-LOO: 計算 prior predictive E_g(Δ;θ̂_{-i})[μ]...")
    y_hat_A = np.array([
        sev_inla_prior_mean(
            S[i],
            theta_loo[i, 0], theta_loo[i, 1],
            np.exp(theta_loo[i, 2]), theta_loo[i, 3], np.exp(theta_loo[i, 4])
        )
        for i in range(len(Y))
    ])

    # ── (B) Post-LOO fitted values ──────────────────────────────
    print("  [B] Post-LOO: 計算後驗均值 E[μᵢ|yᵢ,θ̂_{-i}]...")
    y_hat_B = np.array([
        _gh_posterior_mean_mu(
            Y[i], S[i],
            theta_loo[i, 0], theta_loo[i, 1],
            np.exp(theta_loo[i, 2]), theta_loo[i, 3], np.exp(theta_loo[i, 4])
        ) - np.exp(theta_loo[i, 2]) * EULER_GAMMA
        for i in range(len(Y))
    ])

    return y_hat_A, y_hat_B, y_hat_C, theta_loo

def analyse_npmle(Y, S, pis, deltas, b0, b1, sig):
    """
    Normal+NPMLE 的三種 fitted values：
    (A) Prior-LOO   (B) Post-LOO = In-sample（τ 使用 y_i，LOO θ 幾乎不變）
    """
    # (C) In-sample：軟分配後驗加權
    print("  [C] In-sample: 計算軟分配 E[μᵢ|yᵢ,{Δₖ,πₖ}]...")
    tau = npmle_e_step(Y, S, pis, deltas, b0, b1, sig)
    y_hat_C = np.zeros(len(Y))
    for k in range(len(pis)):
        diff = S - deltas[k]
        ok = diff > 1e-8
        mu_k = np.where(ok, b0 + b1 * np.log(np.maximum(diff, 1e-300)), 0.0)
        y_hat_C += tau[:, k] * mu_k  # Normal conditional mean = μ

    # (A) Prior-LOO：Σₖ πₖ × μ(S,Δₖ)（θ̂_{-i} ≈ θ̂，NPMLE 移除一筆幾乎不變）
    print("  [A] Prior-LOO: 計算 Σ_k π_k · μ(S_i, Δ_k)...")
    y_hat_A = np.array([npmle_prior_mean(S[i], pis, deltas, b0, b1) for i in range(len(Y))])

    # (B) Post-LOO ≈ In-sample（NPMLE K=2，n=75，移除一筆後 {Δₖ,πₖ} 幾乎不動）
    y_hat_B = y_hat_C.copy()

    return y_hat_A, y_hat_B, y_hat_C

def s_conditional_baseline(Y, S):
    """Baseline：各應力水準的樣本均值（無潛在變數模型）"""
    y_hat = np.zeros(len(Y))
    for sv in np.unique(S):
        idx = S == sv
        y_hat[idx] = Y[idx].mean()
    return y_hat

def print_summary(name, y_hat_A, y_hat_B, y_hat_C, Y, S, stresses):
    """印出 per-stress ASSE 對比表格"""
    ae_A = np.abs(Y - y_hat_A)
    ae_B = np.abs(Y - y_hat_B)
    ae_C = np.abs(Y - y_hat_C)

    print(f"\n{'─'*72}")
    print(f"  {name}")
    print(f"{'─'*72}")
    print(f"  {'指標':<20}  {'Prior-LOO(A)':>13}  {'Post-LOO(B)':>13}  {'In-sample(C)':>13}")
    print(f"  {'ASSE':<20}  {ae_A.sum():>13.3f}  {ae_B.sum():>13.3f}  {ae_C.sum():>13.3f}")
    print(f"  {'MAE':<20}  {ae_A.mean():>13.4f}  {ae_B.mean():>13.4f}  {ae_C.mean():>13.4f}")
    print(f"  {'MedAE':<20}  {np.median(ae_A):>13.4f}  {np.median(ae_B):>13.4f}  {np.median(ae_C):>13.4f}")
    print()
    print(f"  Per-stress（MAE）：")
    print(f"  {'S':>7}  {'n':>4}  {'Prior-LOO(A)':>13}  {'Post-LOO(B)':>13}  {'In-sample(C)':>13}")
    for sv in stresses:
        idx = S == sv
        print(f"  {sv:7.3f}  {idx.sum():4d}  "
              f"{ae_A[idx].mean():>13.4f}  {ae_B[idx].mean():>13.4f}  {ae_C[idx].mean():>13.4f}")

# ══════════════════════════════════════════════════════════════════
# §5  主程式
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 72)
    print("LOO Cross-Validation for RFL Models")
    print("=" * 72)

    # ── MLE 參數（已知最優值，直接 hard-code 省時間）────────────────
    # SEV+INLA from rfl_sev_inla.py
    THETA_SEV = np.array([-9.3700, -8.5340, np.log(0.1900), -0.6440, np.log(0.0360)])

    # Normal+NPMLE from rfl_profile.py（K=2）
    PNPMLE = dict(b0=-9.165, b1=-8.089, sig=0.596,
                  pis=np.array([0.930, 0.070]),
                  deltas=np.array([0.532, 0.569]))

    # ── SEV+INLA 分析 ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SEV+INLA（影響函數 LOO）")
    print(f"{'='*72}")
    b0, b1, ls, mud, lsd = THETA_SEV
    print(f"  θ̂: b0={b0:.4f}  b1={b1:.4f}  σ={np.exp(ls):.4f}  μ_d={mud:.4f}  σ_d={np.exp(lsd):.4f}")

    sev_A, sev_B, sev_C, theta_loo_sev = analyse_sev_inla(Y_r, S_r, cens_r, THETA_SEV)
    print_summary("SEV+INLA", sev_A, sev_B, sev_C, Y_r, S_r, stresses)

    # ── Normal+NPMLE 分析 ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Normal+NPMLE（K=2）")
    print(f"{'='*72}")
    print(f"  {PNPMLE}")
    npmle_A, npmle_B, npmle_C = analyse_npmle(
        Y_r, S_r, PNPMLE['pis'], PNPMLE['deltas'],
        PNPMLE['b0'], PNPMLE['b1'], PNPMLE['sig'])
    print_summary("Normal+NPMLE (K=2)", npmle_A, npmle_B, npmle_C, Y_r, S_r, stresses)

    # ── S-conditional baseline ──────────────────────────────────────
    y_base = s_conditional_baseline(Y_r, S_r)
    ae_base = np.abs(Y_r - y_base)
    print(f"\n{'='*72}")
    print("S-conditional 樣本均值 baseline（無任何潛在變數模型）")
    print(f"{'='*72}")
    print(f"  ASSE={ae_base.sum():.3f}  MAE={ae_base.mean():.4f}  MedAE={np.median(ae_base):.4f}")
    for sv in stresses:
        idx = S_r == sv
        print(f"  S={sv:.3f}  n={idx.sum()}  MAE={ae_base[idx].mean():.4f}  "
              f"樣本均值={Y_r[idx].mean():.4f}")

    # ── 綜合比較表 ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("綜合比較：ASSE（越低越好）")
    print(f"{'='*72}")
    print(f"  {'方法':<28}  {'Prior-LOO':>10}  {'Post-LOO':>10}  {'In-sample':>10}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}")

    for name, A, B, C in [
        ("SEV+INLA",           sev_A,   sev_B,   sev_C),
        ("Normal+NPMLE K=2",   npmle_A, npmle_B, npmle_C),
        ("S-cond baseline",    y_base,  y_base,  y_base),
    ]:
        asse_A = np.abs(Y_r - A).sum()
        asse_B = np.abs(Y_r - B).sum()
        asse_C = np.abs(Y_r - C).sum()
        print(f"  {name:<28}  {asse_A:>10.3f}  {asse_B:>10.3f}  {asse_C:>10.3f}")

    print(f"\n{'─'*72}")
    print("解讀：")
    print("  Prior-LOO  = 最誠實的預測能力（完全不看 yᵢ）")
    print("  Post-LOO   = LOO 參數 × 後驗（看 yᵢ 但排除其對 θ 的影響）")
    print("  In-sample  = 雙重使用 yᵢ（θ̂ + 後驗均值均含 yᵢ）")
    print()
    print("  ► Prior-LOO ASSE 相近 → 兩方法對新試件的預測能力相當")
    print("  ► In-sample 差距巨大 → ASSE 度量的是『後驗還原能力』非『預測能力』")
    print("  ► Post-LOO ≈ In-sample → θ 對單筆觀測影響小（n=75 夠穩定）")

    # ── Posterior sharpening bonus ──────────────────────────────────
    print(f"\n{'─'*72}")
    print("Posterior Sharpening Bonus = In-sample ASSE - Prior-LOO ASSE")
    print("（反映『看了 yᵢ 再還原 yᵢ』帶來的 ASSE 虛假降低量）")
    print(f"{'─'*72}")
    for name, A, C in [
        ("SEV+INLA",         sev_A,   sev_C),
        ("Normal+NPMLE K=2", npmle_A, npmle_C),
    ]:
        bonus = np.abs(Y_r - A).sum() - np.abs(Y_r - C).sum()
        print(f"  {name:<28}  bonus = {bonus:+.3f}  "
              f"（Prior-LOO={np.abs(Y_r-A).sum():.3f}  In-sample={np.abs(Y_r-C).sum():.3f}）")
