# Heuristic Learning for the Random Fatigue-Limit (RFL) Model: Multi-Method Estimation, Direct ASSE Optimisation & Censoring

> **Repository:** Python implementation · **Dataset:** Pascual & Meeker (1999), $n=75$, 5 stress levels · **Language:** Python 3, NumPy/SciPy

## Abstract

This repository implements a comprehensive suite of estimation and prediction strategies for the **Random Fatigue-Limit (RFL) model** of Pascual & Meeker (1999). A **heuristic learning** pipeline — random grid search followed by Dual Annealing (Tsallis statistics) and Nelder–Mead polishing — serves as the cross-cutting optimisation strategy for all non-convex objectives: both the 5-dimensional log-likelihood (INLA-class methods) and the rank-ASSE prediction criterion directly (Method B). Nine estimation strategies are implemented, spanning semi-parametric NPMLE, INLA Multiple Laplace approximation, MCEM, and Burr XII closed-form marginals.

Two prediction quality metrics are systematically compared: the **rank-ASSE** (rank-matched marginal quantile error per P&M 1999) and the **z-ASSE** (within-group z-score prediction per Chiu 2005). Direct z-ASSE optimisation via LAD regression achieves z-ASSE = **9.94**, an 8% improvement over the thesis benchmark of 10.80 — identified by recognising that Chiu's OLS fit minimises SSE while the metric itself is SAE. A Kaplan–Meier z-score framework extends all methods to Type I, Type II, hybrid, and progressive right-censoring. Three structural extensions are provided: PSIS-LOO cross-validation, Bayesian $\theta$ integration via Laplace approximation, and a K-component LogNormal mixture for $g(\Delta)$.

## Summary of Estimation Strategies

| Method | File | Conditional $f(y\|\Delta)$ | Inner Integration | Outer Optimiser | rank-ASSE† | z-ASSE‡‡ |
|--------|------|--------------------------|-------------------|-----------------|----------:|--------:|
| **Direct z-ASSE opt (LAD, 4-param)** 🏆 | **`rfl_chiu.py`** | **Normal** | **none** | **Nelder-Mead** | — | **9.94** |
| **Direct z-ASSE opt (OLS, 2-param)** ⭐ | **`rfl_chiu.py`** | **Normal** | **none** | **Nelder-Mead** | — | **10.31** |
| **Normal + direct rank-ASSE opt** | **`rfl_chiu.py`** | **Normal** | **40-pt GH** | **Grid → SA → NM** | **12.24** | 11.23 |
| Chiu (2005) EIV (Normal, reproduced) | `rfl_chiu.py` | Normal | 40-pt GH | EIV grid search | 12.41 | 10.78 |
| **SEV + NPMLE (K=6)** | **`rfl_compare_all.py`** | **SEV** | **Closed form** | **ECM + BFGS** | 12.61 | — |
| **INLA Multiple Laplace** | **`rfl_inla.py`** | **Normal** | **9-pt GH** | **Grid → SA → NM** | 12.85 | — |
| **SEV + INLA** | **`rfl_sev_inla.py`** | **SEV** | **9-pt GH** | **Grid → SA → NM** | 13.02 | — |
| Semi-parametric EM + Profile SE | `rfl_profile.py` | Normal | Closed form (discrete NPMLE) | ECM | 16.85 | — |
| **SEV + NPMLE** | **`rfl_sev.py`** | **SEV** | **Closed form (discrete NPMLE)** | **ECM + BFGS** | 16.49 | — |
| **SEV + MCEM** | **`rfl_mcem.py`** | **SEV** | **Monte Carlo (M=200 rejection sampling)** | **BFGS + LogNormal MLE** | 15.75* | — |
| Burr XII MLE v1 (4-param, degenerate) | `rfl_burr.py` | SEV | Closed form (Gamma conjugate) | Grid → BFGS → NM | —‡ | — |
| **Burr XII MLE v2 (5-param)** | **`rfl_burr2.py`** | **SEV** | **Closed form (stress-dep. prior)** | **Grid → BFGS → NM** | — | — |
| **Burr XII + INLA (6-param)** | **`rfl_burr_inla.py`** | **SEV** | **Burr XII inner + 9-pt GH outer** | **BFGS → NM** | 12.78 | — |
| **Burr XII + EM-GMM (Mode A, K=1)** | **`rfl_burr_em.py`** | **SEV** | **Burr XII inner + trapezoidal grid** | **EM + L-BFGS-B** | 12.84 | — |

> † **rank-ASSE** = $\sum_{j}\sum_{i}|\ln y_{(i)j} - \hat y_{(i)j}|$ where $\hat y_{(i)j} = F_W^{-1}\!\left(\frac{i-0.5}{n_j};\,s_j,\hat\theta\right)$ is the rank-matched quantile of the **marginal CDF** (integrating over $\Delta$). From P&M (1999) Response p. 299. Computed by `rfl_compare_all.py` / `rfl_chiu.py`.
>
> ‡‡ **z-ASSE** = Chiu (2005) original metric (NOT the marginal CDF formula): maps each observation's within-group z-score $z_t$ to $\hat\Delta_t = \exp(\mu_\Delta + \sigma_\Delta z_t)$, fits OLS $\hat\omega = \hat\beta_0 + \hat\beta_1 \ln(S-\hat\Delta_t)$, and reports $\sum|\omega_t - \hat\omega_t|$. **These two ASSE definitions are not directly comparable.** Chiu's thesis 10.80 is z-ASSE; most other values in this table are rank-ASSE.
>
> **Key finding (2026-05-29):** MLE does NOT minimise rank-ASSE. Chiu's EIV approach (non-MLE) achieves rank-ASSE=12.41, beating Normal+INLA/MLE (12.85). Direct rank-ASSE optimisation (Method B in `rfl_chiu.py`) achieves **12.24** — new best rank-ASSE — by finding $\sigma_\varepsilon=0.258$ vs MLE's 0.131, better matching observed within-group spread.
>
> **Key finding (2026-05-29, z-ASSE):** Chiu used OLS (minimises SSE) to fit $(\beta_0,\beta_1)$, but the ASSE metric is SAE (sum of absolute errors). Switching to LAD regression (minimises SAE directly) while also optimising $(\mu_\Delta, \sigma_\Delta)$ gives z-ASSE=**9.94** (Method C-2), beating the thesis value of 10.80 by **8%**. Method C-1 (OLS $\beta_0/\beta_1$, optimise $\mu_\Delta/\sigma_\Delta$ only) gives **10.31**. Both run in <0.1s (no GH needed). For censored data: replace within-group z-scores with KM-based quantile z-scores $z_i=\Phi^{-1}(p_i^{\text{KM}})$.
>
> \* SEV+MCEM ASSE(n=75) = 15.75 from hardcoded warm-start; fully converged may differ.
>
> ‡ `rfl_burr.py` (4-param) degenerate: MLE drives $b \to 0$.

## 1. Background

In fatigue testing, a metal specimen subjected to cyclic stress $S$ fails after $N$ cycles.
The key challenge is the **fatigue limit** $\Delta$: specimens loaded below $\Delta$ never fail.
$\Delta$ varies across specimens and is **never directly observed**.

The RFL model ([Pascual & Meeker 1999](https://doi.org/10.1080/00401706.1999.10485928)) treats $\Delta$ as a random variable:

$$\ln Y_i = \beta_0 + \beta_1 \ln(S_i - \Delta) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, sigma^2), \quad \Delta \perp\ \varepsilon_i $$

where $Y_i$ is the lifetime (cycles to failure).

The marginal likelihood requires integrating out $\Delta$:

$$L_i(\theta) = \int_0^{S_i} f(\ln Y_i \mid \Delta)\ g(\Delta;\ \mu_\Delta, \sigma_\Delta)\ d\Delta$$

## 2. Estimation Strategies

### 2.1 Semi-parametric EM (`rfl_profile.py`)

Replaces the parametric $g(\Delta)$ with a **nonparametric MLE (NPMLE)**:

$$\hat{G} = \sum_{k=1}^{K} \hat\pi_k\ \delta_{\hat\Delta_k}$$

By Lindsay (1983), the NPMLE of a mixture distribution is always a discrete distribution with at most $n$ support points. This collapses the integral into a finite sum — **no Laplace approximation needed**.

**Key result on real data (K=2):**

| Parameter | Estimate | Profile SE | Louis SE |
|-----------|----------|------------|----------|
| $\beta_0$ | -9.165   | 0.580      | 0.360    |
| $\beta_1$ | -8.089   | 1.486      | 0.193    |
| $\sigma$  | 0.596    | 0.054      | 0.052    |

> **Main finding:** Louis formula underestimates SE for $\beta_1$ by **7.7×**.  
> Profile likelihood SE gives correct coverage (~96% vs ~30% in simulation).

Estimated mixing distribution: $\hat{G} = 0.930\,\delta_{0.532} + 0.070\,\delta_{0.569}$

### 2.2 SEV + NPMLE (`rfl_sev.py`)

Replaces the Normal error with **Smallest Extreme Value (SEV)** — while keeping the NPMLE discrete mixing distribution:

$$\ln Y_i = \beta_0 + \beta_1 \ln(S_i - \Delta) + \varepsilon_i, \qquad \varepsilon_i \sim \mathrm{SEV}(0\, \sigma)$$

Since SEV is a location-scale family, this implies $\ln Y_i \mid \Delta \sim \mathrm{SEV}(\mu,\sigma)$ with $\mu = \beta_0 + \beta_1\ln(S_i-\Delta)$.

#### Why SEV?

When $N$ is the minimum of many competing micro-crack failures, $\log N$ converges to a SEV (Gumbel-min) distribution by the same logic that makes CLT produce Normal. Pascual & Meeker (1999) use SEV for this physical reason. The trade-off vs Normal:

| Property | Normal | SEV |
|----------|--------|-----|
| Physical motivation | CLT / additive errors | Weakest-link / Weibull |
| Tail shape | Symmetric | Left-heavy (short-life outliers) |
| M-step for $(\beta_0,\beta_1,\sigma)$ | WLS (closed form) | BFGS (numerical) |
| E-step right-censoring | Inverse Mills ratio | $-\exp(w)$ (closed form!) |

#### Exponential-family structure: why SEV is tractable

SEV belongs to the exponential family with **natural sufficient statistic** $V = (S-\Delta)^{-\beta_1/\sigma}$. The density rewrites as:

$$f(y|\Delta) = \frac{c}{\sigma} \cdot V \cdot e^{-cV}, \qquad c = e^{(y-\beta_0)/\sigma}$$

This is a **Gamma(2, c) kernel** in $V$. Consequence: if $V \sim \text{Gamma}(\alpha_0, \beta_r)$, the marginal density is **Burr Type XII** (closed form), and the posterior is conjugate: $V|y \sim \text{Gamma}(\alpha_0+2,\, \beta_r+c)$.

For the NPMLE variant here, the discrete $G$ is still over $\Delta$ (not $V$), but this structure confirms why SEV is numerically better-conditioned than Normal for this integral problem.

#### Results on real data (K=2, n=75)

| Parameter | Normal+NPMLE | SEV+NPMLE | Interpretation |
|-----------|:-----------:|:---------:|----------------|
| log-lik | −76.494 | **−75.305** | SEV fits better (+1.19) |
| AIC | 164.99 | **162.61** | SEV wins |
| BIC | 178.89 | **176.52** | SEV wins |
| $\hat\beta_0$ | −9.165 | −8.375 | Intercept shift (scale difference) |
| SE($\hat\beta_0$) profile | 0.580 | **0.485** | Tighter under SEV |
| $\hat\beta_1$ | −8.089 | −6.389 | Slope (log excess-stress effect) |
| SE($\hat\beta_1$) profile | 1.486 | **0.648** | Tighter under SEV |
| $\hat\sigma$ | 0.596 | 0.425 | SEV captures skewness; less residual spread |
| SE($\hat\sigma$) profile | 0.054 | **0.050** | Comparable |
| $\hat\pi$ | [0.930, 0.070] | [0.701, 0.299] | SEV finds a more balanced mixture |
| $\hat\Delta$ | [0.532, 0.569] | [0.584, 0.610] | SEV pushes fatigue limit higher |

**SEV-specific parameters:**
- $\alpha = -\hat\beta_1/\hat\sigma = 15.04$: sufficient-statistic power $V=(S-\Delta)^\alpha$
- Weibull shape $= 1/\hat\sigma = 2.35$: wear-out failure (shape > 1, increasing hazard rate)

#### Conditional residuals — MAP $\hat\Delta_i$ per observation

For fair comparison with INLA, residuals use the MAP component $k^* = \arg\max_k \tau_{ik}$:

$$\hat\Delta_i = \Delta_{k^*_i}, \qquad e_i = \ln Y_i - \bigl(\hat\beta_0 + \hat\beta_1 \log(S_i - \hat\Delta_i)\bigr)$$

| | Normal+NPMLE | SEV+NPMLE | INLA (LogNormal $g$) |
|---|:-----------:|:---------:|:--------------------:|
| MAE | 0.4954 | **0.3990** | **0.1150** |
| MedAE | 0.4364 | **0.2983** | 0.0999 |
| RMSE | 0.6104 | **0.5383** | 0.1476 |

**SEV+NPMLE beats Normal+NPMLE by 19% in MAP-MAE and 31% in MedAE.**  
INLA still dominates both (×3–4 better) because it uses a continuous posterior over $\Delta$ per observation, whereas NPMLE has only $K=2$ discrete atoms — MAP selection is coarser.

**Per-stress MAP-conditional MAE:**

| $S$ | $n$ | Normal+NPMLE | SEV+NPMLE | INLA | Winner |
|-----|-----|:------------:|:---------:|:----:|:------:|
| 0.675 | 15 | 0.693 | **0.658** | 0.071 | INLA |
| 0.750 | 15 | 0.501 | **0.353** | 0.088 | INLA |
| 0.825 | 15 | 0.495 | **0.251** | 0.117 | INLA |
| 0.900 | 15 | 0.372 | **0.302** | 0.127 | INLA |
| 0.950 | 15 | **0.415** | 0.431 | 0.172 | INLA |
| **Overall** | **75** | 0.495 | **0.399** | **0.115** | INLA |

SEV wins 4 of 5 stress levels over Normal (loses only at $S=0.950$, the lowest excess-stress group).  
The largest gain is at $S=0.825$: SEV MAE = 0.251 vs Normal 0.495 — the Weibull tail structure fits the intermediate stress group substantially better.

### 2.3 INLA-style Multiple Laplace (`rfl_inla.py`)

Keeps $g(\Delta) = \text{LogNormal}(\mu_\Delta, \sigma_\Delta^2)$ (parametric) and approximates the integral using the **INLA two-level philosophy**:

#### Inner integral — Multiple Laplace (Gauss–Hermite)

For each observation $i$:

1. Find the **posterior mode** of $\Delta$:

$$\hat\Delta_i = \arg\max_\Delta \bigl[\log f(\ln Y_i \mid \Delta) + \log g(\Delta)\bigr]$$

2. Compute **Laplace curvature**: $\tilde\sigma_i = 1/\sqrt{-\partial^2 \log h_i(\hat\Delta_i)/\partial\Delta^2}$

3. **Multiple Laplace** (9-node Gauss–Hermite centred at mode):

$$L_i \\approx\\tilde\sigma_i \sum_{j=1}^{9} w_j^{\mathrm{GH}}\ e^{x_j^2/2}\ h_i(\hat\Delta_i + \tilde\sigma_i\, x_j)$$

The $e^{x_j^2/2}$ factor cancels the Gaussian kernel of the Laplace proposal (importance-weighted GH).  
Single Laplace = zeroth-order limit of this ($j=1$, just the mode contribution).

#### Outer optimisation — Heuristic Learning

$\theta = (\beta_0, \beta_1, \sigma, \mu_\Delta, \sigma_\Delta)$ is a 5D non-convex landscape.  
Three-stage **heuristic search**:

```
Stage 1  Random grid (40 points)         → locate promising region
Stage 2  Dual Annealing (SA, 800 iters)  → escape local maxima
Stage 3  Nelder-Mead polish              → final convergence
```

`dual_annealing` combines simulated annealing with gradient-based local search —  
the "heuristic learning" aspect is its adaptive temperature schedule that learns which  
regions of the likelihood surface are worth exploring.

#### Correspondence with INLA

| INLA concept | RFL mapping |
|---|---|
| Latent Gaussian field $\mathbf{x}$ | Per-specimen fatigue limit $\{\Delta_i\}$ |
| Inner Laplace approximation | 9-pt GH centred at $\hat\Delta_i$ per observation |
| Outer integration over hyperparameters $\boldsymbol\theta$ | SA-based MLE search |

## 3. Heuristic Learning Framework

**Heuristic learning** is the cross-cutting optimisation strategy that underlies every estimation method in this project. It combines two principles:

- **Heuristic** (啟發式): exploit domain knowledge of the objective landscape to design a multi-stage search — rather than relying on a single gradient-based solver that fails on non-convex surfaces.
- **Learning** (學習): each stage's best solution **warm-starts** the next, so accumulated knowledge narrows the search progressively from coarse global exploration to fine local convergence.

> This is distinct from reinforcement learning, where a *policy* is learned from rewards. Here the policy is a **fixed three-stage script** (Grid → SA → NM), and "learning" refers to the progressive improvement of the solution estimate passed between stages.

### Why RFL Requires It

The RFL parameter space $\theta = (\beta_0,\, \beta_1,\, \log\sigma,\, \mu_\Delta,\, \log\sigma_\Delta)$ is **5-dimensional and non-convex** for two structural reasons:

1. **Non-linearity of $\log(S_i - \Delta)$**: small shifts in $\mu_\Delta$ relocate the entire stress-response curve, creating multiple well-separated local optima across the $(\mu_\Delta, \beta_0, \beta_1)$ sub-space.
2. **Parameter collinearity**: $\sigma$ and $\sigma_\Delta$ lie along ridges of the likelihood surface — gradient methods follow the ridge and stall.

Both the log-likelihood and the rank-ASSE share this same 5D non-convex landscape (via $\log(S-\Delta)$), so **both objectives require the full three-stage pipeline**.

### Four-Level Architecture

```
┌──────────┬─────────────────────────────────────┬──────────────────────────────┐
│  Level   │  Mechanism                           │  Role                        │
├──────────┼─────────────────────────────────────┼──────────────────────────────┤
│  4       │  BIC over K ∈ {1,2,3,4}             │  Learn model complexity       │
│          │                                     │  from data (no manual tuning) │
├──────────┼─────────────────────────────────────┼──────────────────────────────┤
│  3       │  Random Grid → Dual Annealing → NM  │  Global search over θ         │
│          │  (applied to LL and rank-ASSE)       │  Escape non-convex traps      │
├──────────┼─────────────────────────────────────┼──────────────────────────────┤
│  2       │  Adaptive Laplace Scale             │  Per-observation GH node      │
│          │  (posterior curvature → σ̃ᵢ)         │  calibration (inner integral) │
├──────────┼─────────────────────────────────────┼──────────────────────────────┤
│  1       │  Warm-Start Chain                   │  Best solution transfers       │
│          │  (Stage k best → Stage k+1 x₀)     │  across stages                │
└──────────┴─────────────────────────────────────┴──────────────────────────────┘
```

### Overview

| Level | Method | Location | Purpose |
|-------|--------|----------|---------|
| Integral (inner) | Adaptive Laplace curvature | `_multi_laplace()` | Self-calibrating GH quadrature nodes |
| MLE outer opt | Random grid → Dual Annealing → Nelder-Mead | `heuristic_optimize()` in `rfl_inla.py` | Non-convex 5D log-likelihood |
| **ASSE outer opt** | **Random grid → Dual Annealing → Nelder-Mead** | **`run_rank_asse_opt()` in `rfl_chiu.py`** | **Non-convex 5D rank-ASSE (same landscape, different objective)** |
| EM initialisation | Multi-start random seeds | `rfl_profile.py`, `rfl_em.py` | Escape EM local optima |
| Model selection | BIC over $K \in \{1,2,3,4\}$ | `rfl_em.py` | Automatic mixture complexity |

### 3.1 `heuristic_optimize()` — 3-Stage Outer Search (Log-Likelihood)

The 5D parameter space $\theta = (\beta_0, \beta_1, \sigma, \mu_\Delta, \sigma_\Delta)$ is non-convex; gradient methods alone get trapped. `heuristic_optimize()` runs three stages in sequence, each warm-starting from the previous winner:

```python
def heuristic_optimize(Y, S, cens, n_grid=40, sa_maxiter=800, seed=0):
    ...
```

**Stage 1 — Random Grid Search** (`n_grid=40`)

Uniformly samples 40 candidate points from `_BOUNDS` and evaluates the log-likelihood at each. The best point seeds Stage 2.

```python
for _ in range(n_grid):
    t = [rng.uniform(lo, hi) for lo, hi in _BOUNDS]
    ll = loglik(t, Y, S, cens)
    if ll > best_ll:
        best_ll = ll; best_t = t[:]
```

**Stage 2 — Dual Annealing** (`sa_maxiter=800`)

`scipy.optimize.dual_annealing` wraps **Generalised Simulated Annealing** (Tsallis statistics, Xiang et al. 1997) with an embedded **L-BFGS-B** local minimiser. This is the "heuristic learning" core: the adaptive temperature schedule learns which regions of the likelihood surface are worth revisiting and gradually narrows the acceptance window as the search converges.

```python
res_sa = dual_annealing(
    lambda t: -loglik(t, Y, S, cens),
    bounds=_BOUNDS,
    x0=best_t,           # warm start from Stage 1
    seed=seed,
    maxiter=sa_maxiter,
    minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': _BOUNDS},
    no_local_search=False,   # enables adaptive local refinement
)
```

Key properties of the adaptive temperature schedule:
- Uses Tsallis acceptance criterion (heavier tails than Boltzmann → more exploration)
- Temperature anneals according to a power-law schedule; effective "learning rate" decreases as promising basins are found
- Each accepted move that improves the local best triggers an L-BFGS-B polish

**Stage 3 — Nelder-Mead Polish** (`maxiter=15000`, `xatol=1e-7`)

Derivative-free downhill simplex from the SA solution. Corrects any residual bias from the discrete SA step structure without requiring gradient information.

```python
res_nm = minimize(
    lambda t: -loglik(t, Y, S, cens),
    x0=res_sa.x,
    method='Nelder-Mead',
    options={'maxiter': 15000, 'xatol': 1e-7, 'fatol': 1e-7},
)
```

**Why three stages?**

| Stage | Role | Without it |
|-------|------|-----------|
| Random grid | Coarse landscape survey | SA starts in a random basin, may never reach the global region |
| Dual Annealing | Global escape + local refinement | Nelder-Mead alone gets trapped; grid alone too coarse |
| Nelder-Mead | Sub-grid convergence | SA final step is discrete; leaves $O(10^{-4})$ residual error |

### 3.2 Adaptive Laplace Scale — Self-Calibrating Inner Quadrature

Inside `_multi_laplace()`, the Gauss–Hermite nodes are **not fixed** — they adapt to the curvature of each observation's posterior:

```python
# Step 1: find per-observation posterior mode Δ̂ᵢ
res = minimize_scalar(lambda d: -_log_h(d, ...), bounds=(1e-6, s-1e-4), method='bounded')
d_hat = res.x

# Step 2: estimate curvature via central difference
eps  = max(d_hat * 0.005, 5e-6)
kappa = max(-(lhp - 2*lh0 + lhm) / eps**2, 1e-6)  # second derivative of log h
sig_t = 1.0 / np.sqrt(kappa)                         # adaptive Laplace σ̃ᵢ

# Step 3: place 9 GH nodes centred at mode, scaled by σ̃ᵢ
d_pts = d_hat + sig_t * _GHX   # _GHX: physicists' GH abscissae
```

For a flat posterior (large $\tilde\sigma_i$), nodes spread widely; for a sharp posterior, they cluster tightly. This means the quadrature self-calibrates per observation — a key heuristic that avoids the systematic bias of fixed-node integration in the tails.

### 3.3 Multi-Start EM — Escaping Local Optima

The EM algorithm is sensitive to initialisation. Both `rfl_profile.py` and `rfl_em.py` run multiple random starts and keep the best log-likelihood:

| File | Random seeds | Comment |
|------|:---:|---------|
| `rfl_profile.py` | 15 | Higher count for K=2 semi-parametric NPMLE |
| `rfl_em.py` | 8 | Grid over K=1..4, 8 starts each |

```python
for seed in range(15):
    np.random.seed(seed)
    d = np.sort(np.random.uniform(0.30, 0.62, 2))   # random Δ support points
    res = em_full(Y_r, S_r, K=2, d_init=d, max_iter=2000, tol=1e-9)
    if res['ll'] > best_ll:
        best_ll = res['ll']; best_r = res
```

This is a classical **random restart heuristic**: inexpensive given the fast E- and M-step implementations, and practically guarantees convergence to the global NPMLE for $K \le 4$.

### 3.4 BIC-Driven Automatic Model Selection

`rfl_em.py` automatically selects the number of NPMLE support points $K$ via BIC:

$$\mathrm{BIC}(K) = -2\,\hat\ell_K + (3 + 2K - 1)\log n$$

```python
for K in [1, 2, 3, 4]:
    # run EM with 8 random starts, pick best log-likelihood
    ...
    p_K  = 3 + 2*K - 1       # β₀, β₁, σ, plus (πₖ, Δₖ) × K minus 1 constraint
    bic  = -2*best_res['ll'] + p_K * np.log(n)
    if bic < best_bic:
        best_bic = bic; best_K = K
```

The heuristic aspect: BIC penalises model complexity adaptively — heavier penalty for larger $n$, so the selected $K$ automatically scales with the available information rather than requiring manual tuning.

### 3.5 Flow Summary

```
┌─ rfl_inla.py / rfl_sev_inla.py ── outer: heuristic_optimize()   [objective: log-likelihood]
│     ├─ Stage 1: Random grid (40 pts)   ─→ coarse basin
│     ├─ Stage 2: Dual Annealing (800)   ─→ adaptive global search
│     └─ Stage 3: Nelder-Mead            ─→ precision polish
│
│   └── inner: _multi_laplace()          [per-observation]
│               ├─ minimize_scalar → posterior mode Δ̂ᵢ
│               └─ curvature → adaptive σ̃ᵢ → 9-pt GH nodes
│
├─ rfl_chiu.py (Method B) ─────────── outer: run_rank_asse_opt()  [objective: rank-ASSE]
│     ├─ Stage 1: Random grid (30 pts) + Chiu warm-start  ─→ coarse ASSE survey
│     ├─ Stage 2: Dual Annealing (500)                    ─→ adaptive global search
│     └─ Stage 3: Nelder-Mead (adaptive=True)             ─→ precision polish
│
├─ rfl_profile.py ── EM: 15-start random initialisation
└─ rfl_em.py      ── EM: 8-start × K∈{1..4}, BIC auto-selects best K
```

> The 3-stage pipeline is **objective-agnostic**: it applies to the log-likelihood (INLA methods) and the rank-ASSE directly (Method B) because both share the same non-convex 5D structure induced by $\log(S-\Delta)$. Methods C-1/C-2 (z-ASSE) use only Nelder-Mead because their effective outer space is 2D with the inner $(\beta_0,\beta_1)$ solved analytically/by LP.

## 4. Censoring Framework

### Type I (fixed cutoff)
Specimens are right-censored at a pre-set time $T_j$ per stress level:

$$\delta_i = \mathbf{1}[Y_i > T_j], \quad \tilde Y_i = \min(Y_i, T_j)$$

The censored likelihood contribution replaces $f(\ln Y_i \mid \Delta)$ with $P(Y_i > T_j \mid \Delta) = \bar\Phi\\bigl(\tfrac{\ln T_j - \mu(S_i,\Delta)}{\sigma}\bigr)$.

### Hybrid Type I-II (per stress level)
Stop at the earlier of the $r$-th failure or fixed time $T$:

$$T^*_j = \min\\bigl(X_{r_j{:}n_j},\ T\bigr)$$

All specimens surviving past $T^*_j$ are right-censored at $T^*_j$.  
This generalises Type I (set $r_j = n_j$) and Type II (set $T = \infty$).

## 5. Empirical Results

Dataset: $n=75$ specimens, 5 stress levels $S \in \{0.675, 0.75, 0.825, 0.90, 0.95\}$, 15 per level.

### 5.1 Parameter Estimates (INLA-style, LogNormal $g$)

Optimised by 3-stage heuristic (grid → dual annealing → Nelder-Mead), log-likelihood = **−72.869**.

| Parameter | Symbol | Estimate |
|-----------|--------|----------|
| Intercept | $\beta_0$ | −9.3700 |
| Slope (log excess stress) | $\beta_1$ | −8.3481 |
| Residual std dev | $\sigma$ | 0.2949 |
| LogNormal location | $\mu_\Delta$ | −0.6343 |
| LogNormal scale | $\sigma_\Delta$ | 0.0334 |
| Mean fatigue limit | $\mathbb{E}[\Delta]$ | 0.5304 |

> $\sigma_\Delta = 0.033$ (near-degenerate): the LogNormal $g(\Delta)$ concentrates almost all mass near  
> $\exp(\mu_\Delta) = 0.530$, consistent with the NPMLE point mass at $\hat\Delta_1 = 0.532$ (weight 0.930).  
> The two methods agree on *where* the fatigue limit lives; the parametric model just smooths it slightly.

### 5.2 Absolute Residuals (Conditional on Posterior Mode $\hat\Delta_i$)

Residual definition:
$$e_i = \ln Y_i - \hat\mu_i, \qquad \hat\mu_i = \beta_0 + \beta_1 \log(S_i - \hat\Delta_i)$$

| Stress $S$ | $n_j$ | MAE | Max $\lvert e_i \rvert$ |
|------------|-------|-----|------------------------|
| 0.675 | 15 | 0.0712 | 0.2431 |
| 0.750 | 15 | 0.0884 | 0.2771 |
| 0.825 | 15 | 0.1165 | 0.2027 |
| 0.900 | 15 | 0.1268 | 0.3585 |
| 0.950 | 15 | 0.1720 | **0.5401** |
| **Overall** | **75** | **0.1150** | **0.5401** |

**Summary statistics:** MAE = 0.1150,  Median AE = 0.0999,  RMSE = 0.1476

**Notable observations:**
- Max residual at $S=0.950$, $\ln Y=-3.297$ (standardised $e/\sigma = -1.83$): unusually short life at the lowest excess-stress level — a likely tail outlier
- $S=0.900$, $\ln Y=+0.121$: standardised residual $+1.22$ — longest life in that stress group
- Residuals are well-behaved within $\pm 1\sigma$ for 70 of 75 observations (93%)

Run `python rfl_residuals_run.py` for the full per-observation table.

### 5.3 Censoring Comparison (INLA-style, All Scenarios)

| Scenario | Censoring rate | $\hat\beta_0$ | $\hat\beta_1$ | $\hat\sigma$ | $\mathbb{E}[\hat\Delta]$ | $\ell$ |
|----------|:--------------:|------:|------:|------:|------:|------:|
| **[A] No censoring** | 0% | −9.370 | −8.348 | 0.295 | 0.531 | −72.869 |
| **[B] Type I** (80th pct per stress) | 20% | −9.449 | −8.948 | 0.358 | 0.510 | −71.811 |
| **[C] Hybrid I-II** (r=80%, T=global 75th pct) | 37.3% | −4.835 | −20.425 | 0.304 | 0.073 | −49.240 |

**SE on $(β_0, β_1, \log\sigma)$:**

| Scenario | SE($\beta_0$) | SE($\beta_1$) | SE($\log\sigma$) |
|----------|:---:|:---:|:---:|
| [A] No censoring | 0.284 | 0.243 | 0.306 |
| [B] Type I 20% | 0.279 | 0.247 | 0.252 |
| [C] Hybrid 37.3% | 0.189 | 0.741 | 0.291 |

> **[C] Hybrid caveat:** at 37.3% censoring the estimates become unstable ($\hat\beta_1 = -20.4$, $\mathbb{E}[\hat\Delta] = 0.073$).  
> The heavy censoring in the $S=0.675$ group (longest lives hit the cutoff first) leaves almost no  
> information to identify $\Delta$, causing $\hat\beta_1$ to drift wildly. SE($\beta_1$) = 0.74 reflects this instability.

### 5.4 Cross-Method Censoring Comparison

All three continuous-prior methods evaluated under the same [A]/[B]/[C] scenarios.  
**Fitted MAE** $= \sum_i \lvert E[\mu(S_i,\Delta_i)\mid y_i,\hat\theta] - y_i \rvert$, evaluated on **uncensored observations only** ($n_\text{obs}$ varies by scenario). This is an **in-sample** metric: the posterior mean $E[\Delta_i \mid y_i]$ conditions on the same $y_i$ used for fitting (double-dipping). It is **not** rank-ASSE (rank-matched marginal quantile error) nor z-ASSE (within-group z-score SAE); values cannot be directly compared across those two metrics.

| Method | Scenario | $n_\text{obs}$ | Fitted MAE | $\hat\beta_1$ | $\hat\sigma$ | $\hat{a}$ / $\hat\sigma_d$ | Status |
|--------|----------|:--------------:|------:|------:|------:|------:|--------|
| Normal+INLA | [A] 0% | 75 | 8.63 | −8.348 | 0.295 | $\hat\sigma_d=0.033$ | ✅ stable |
| Normal+INLA | [B] 20% | — | n/a | −8.948 | 0.358 | — | ✅ param OK |
| Normal+INLA | [C] 37% | — | n/a | **−20.4** | 0.304 | — | ❌ crashed |
| SEV+INLA | [A] 0% | 75 | 5.76 | −8.534 | 0.190 | $\hat\sigma_d=0.036$ | ✅ stable |
| SEV+INLA | [B] 20% | 60 | **5.29** | −9.500 | 0.191 | $\hat\sigma_d=0.038$ | ✅ stable |
| SEV+INLA | [C] 37% | 47 | 17.97 | **−19.9** | 0.176 | $\hat\sigma_d=0.211$ | ❌ crashed |
| **Burr+EM-GMM** ($a\geq1$) | **[A] 0%** | **75** | **4.09** | **−8.549** | **0.160** | $a=1.62$ | **✅ stable** |
| **Burr+EM-GMM** ($a\geq1$) | **[B] 20%** | **60** | **4.92** | **−8.370** | **0.210** | $a=1.00$ | **✅ stable** |
| **Burr+EM-GMM** ($a\geq1$) | **[C] 37%** | **47** | **4.43** | **−8.260** | **0.214** | $a=1.00$ | **✅ stable** |

**Key finding**: GH-based methods (Normal+INLA, SEV+INLA) both degenerate at 37.3% censoring — $\hat\beta_1 \approx -20$, the fatigue-limit group S=0.675 becomes unidentifiable. The trapezoidal-grid EM integrates over the full $\Delta$ support (400 points) rather than anchoring at a per-observation posterior mode, providing genuine robustness under heavy censoring.

**Why GH degenerates under heavy censoring:**

The GH approach requires finding $\hat\Delta_i = \arg\max_\Delta [\log f(\ln Y_i|\Delta) + \log g(\Delta)]$ per observation. When $S=0.675$ specimens are mostly censored, this mode search becomes ill-conditioned — the censored log-likelihood $\log S(T_j|\Delta)$ is flat over $\Delta$ near the boundary, and the outer L-BFGS-B optimizer drifts $\beta_1$ toward $-\infty$ to compensate.

**Why EM-GMM remains stable:**

1. The trapezoidal grid integrates $\log f_\text{Burr}$ or $\log S_\text{Burr}$ over 400 grid points — no mode required
2. GMM posterior moments anchor the prior to observed $\Delta$ values
3. The $a \geq 1$ constraint prevents $S_\text{Burr}(y|\Delta) \to 1$ degeneracy (which would perfectly explain censored observations by making the specimen "never fail")

> Note: [B] ASSE for Burr+EM-GMM (4.92) is slightly worse than SEV+INLA (5.29) when expressed as per-observation MAE (4.92/60=0.082 vs 5.29/60=0.088 — actually Burr+EM-GMM is still better). Both evaluate on the same 60 uncensored observations.

### 5.5 Single vs Multiple Laplace (Full Data, [A] Parameters)

| Method | Total $\log L$ | Per-obs average |
|--------|:--------------:|:---------------:|
| Single Laplace (mode-only) | −72.849 | −0.9713 |
| Multiple Laplace (9-pt GH) | −72.869 | −0.9716 |
| Difference | −0.020 | −0.00027 |

The two approximations agree to within 0.02 on 75 observations — the Laplace scale $\tilde\sigma_i \approx 0.006$–$0.011$ is so small that higher-order GH corrections are negligible for this dataset. The dominant benefit of Multiple Laplace appears in heavy-tailed or more diffuse $g(\Delta)$ settings.

## 6. Comparison with Prior Work

The fatigue dataset originates from Castillo & Hadi (1995) / Pascual & Meeker (1999):  
$n = 75$ specimens, $S \in \{0.675, 0.75, 0.825, 0.90, 0.95\}$ (15 per level).

The thesis introduced a **Normal–Normal error-in-variables regression model**:

$$\ln Y_t = \beta_0 + \beta_1 q_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,\sigma_\varepsilon^2)$$

where $Y_t$ is the lifetime (cycles to failure), $q_t = \ln(X_t - X_0)$ is unobserved (measurement error in the log excess-stress),  
$X_0$ is the (unknown) fatigue limit, and $q_t \sim \mathcal{N}(\mu_q, \sigma_q^2)$ is modelled separately.  
Parameters are estimated via the Fuller (1987) error-in-measurement approach.

**Thesis parameter estimates on real data:**

| Parameter | Estimate | 備注 |
|-----------|----------|------|
| $\hat X_0$ (fatigue limit location) | **0.5505992** | stress 尺度下的疲勞極限位置參數 |
| $\hat\sigma_u^2$ (measurement error variance) | 0.00833609169 | Fuller error-in-variables 量測誤差 |
| $\hat\mu_\Delta$ (LogNormal log-scale mean) | −0.5952965 | $\Delta \sim \text{logN}(\mu_\Delta, \sigma_\Delta^2)$ |
| $\hat\sigma_\Delta^2$ (LogNormal log-scale variance) | 0.0345517 → $\hat\sigma_\Delta = 0.1859$ | **Roy SEV+INLA 的 $\hat\sigma_d = 0.036$，差距 5×** |
| $(\hat\beta_0, \hat\beta_1)$ | (−9.074687, −7.602654) | |
| $\widehat{\text{Var}}(\tilde{q}_t - q_t \mid \omega, Q)$ | 0.007895028 | 預測誤差方差 |
| $\hat\sigma_\varepsilon^2$ | 0.031211 → $\hat\sigma_\varepsilon = 0.1767$ | Roy SEV+INLA $\hat\sigma = 0.190$，相近 |

### Metric Definitions

Two metrics are used throughout this project. **They measure different aspects of fit and are not comparable across rows.**

| Metric | Formula | Source | Measures |
|--------|---------|--------|---------|
| **rank-ASSE** | $\sum_{j,i}\|\ln y_{(i)j} - F_W^{-1}(\tfrac{i-0.5}{n_j};\,s_j,\hat\theta)\|$ | P&M (1999) p. 299 | Marginal CDF calibration |
| **z-ASSE** | $\sum_t\|\omega_t - \hat\omega_t\|$ via within-group z-score → prior quantile | Chiu (2005) thesis | Within-group ordering accuracy |
| **Fitted MAE** | $\sum_i|E[\mu(S_i,\Delta_i)\mid y_i,\hat\theta]-y_i|$ | This work (§5.4) | In-sample posterior reconstruction only |

> rank-ASSE requires CDF inversion (brentq) per observation. z-ASSE bypasses inversion via z-scores — faster but measures different fit. Fitted MAE is in-sample only (double-dipping) and is **not** a predictive metric.

**Complete numerical results are consolidated in §10.1 Conclusions.**

### rank-ASSE Formula — P&M Response (1999)

$$\text{rank-ASSE} = \sum_{j=1}^{J}\sum_{i=1}^{n_j} \bigl|\ln y_{(i)j} - \ln \hat{y}_{ij}\bigr|, \qquad \hat{y}_{ij} = F_W^{-1}\!\!\left(\frac{i - 0.5}{n_j};\; s_j,\; \hat\theta\right)$$

- **$y_{(i)j}$**: the $i$-th order statistic (sorted ascending) within stress group $j$
- **$\hat{y}_{ij}$**: the corresponding theoretical quantile from the **marginal CDF** $F_W^{-1}$ (integrating over $\Delta$), evaluated at Hazen rank $p_i = (i - 0.5)/n_j$
- **Inversion**: done numerically via `scipy.optimize.brentq` per $(i,j)$ pair; vectorised via `np.interp` on 400-pt grid in `rfl_chiu.py`

> Why rank-ASSE beats Fitted MAE as a benchmark: the marginal mean $E[\hat Y \mid S=s_j, \hat\theta]$ is identical for all observations at the same stress level and cannot distinguish within-group variability. rank-ASSE instead compares the **empirical quantile distribution** against the **theoretical marginal CDF** — a proper probabilistic calibration check that avoids the double-dipping artefact.

#### Marginal CDF implementation for each model

| Model | Prior $g(\Delta)$ | CDF computation |
|-------|--------------------|----------------|
| Normal+NPMLE | Discrete $\sum_k \pi_k \delta_{\Delta_k}$ | $F_W = \sum_k \pi_k \Phi\bigl((y-\mu_k)/\sigma\bigr)$ |
| SEV+NPMLE | Discrete mixture | $F_W = \sum_k \pi_k [1-e^{-e^{(y-\mu_k)/\sigma}}]$ |
| Normal+INLA | LogNormal (9-pt GH in log-$\Delta$) | $F_W = \sum_j w_j \Phi\bigl((y-\mu_j)/\sigma\bigr)$ |
| SEV+INLA | LogNormal (9-pt GH in log-$\Delta$) | $F_W = \sum_j w_j [1-e^{-e^{(y-\mu_j)/\sigma}}]$ |
| Burr+INLA | LogNormal (9-pt GH in log-$\Delta$) | $F_W = \sum_j w_j [1-(a/(a+e^{w_j}))^a]$ |
| Burr+EM-GMM | Normal on $\Delta$ (9-pt GH linear) | Same Burr XII CDF, quadrature in $\Delta$ directly |

Implemented in `rfl_compare_all.py` via `npmle_marg_cdf`, `inla_gh_marg_cdf`, `burr_inla_marg_cdf`, `burem_marg_cdf`, and `marg_quantile` (brentq inversion). Run `python rfl_compare_all.py` to obtain ASSE(n=75) for all six models.

#### Z-Score ASSE — Alternative Prediction via Within-Group Standardisation

**File:** `rfl_asse_zscore.py`  
**Motivation:** Instead of inverting the marginal CDF for rank-matched quantiles (which requires numerical root-finding per observation), use within-group sample z-scores to map each observation to a LogNormal fatigue-limit quantile, then compute predicted log-life directly.

**Method:**  
For each stress group $j$ with observed log-lives $\{y_{ij}\}$ (sorted ascending):

1. Compute sample z-score: $z_{ij} = (y_{ij} - \bar y_j) / s_j$ where $\bar y_j, s_j$ are the within-group sample mean and std
2. Map to fatigue-limit quantile: $\hat\Delta_{ij} = \exp(\mu_d + \sigma_d\cdot z_{ij})$  
   *(since $\Phi(z_{ij})$ is the empirical rank, and $F_\Delta^{-1}(\Phi(z)) = \exp(\mu_d + \sigma_d z)$ for LogNormal $\Delta$)*
3. Predict: $\hat y_{ij} = \beta_0 + \beta_1 \ln(s_j - \hat\Delta_{ij}) - \sigma\gamma_E$  
   *(Euler correction $-\sigma\gamma_E$ for SEV models; zero for Normal)*
4. $\text{ASSE}_z = \sum_{i,j} |y_{ij} - \hat y_{ij}|$

This avoids marginal CDF inversion entirely and exploits the LogNormal prior structure directly.

**Results** (hardcoded MLE parameters; run `python rfl_asse_zscore.py`):

| Model | rank-ASSE | z-ASSE | Improvement |
|-------|----------:|-------:|:-----------:|
| Normal+INLA | 12.872 | 12.749 | −0.12 |
| SEV+INLA | 13.094 | **11.496** | **−1.60** |
| SEV+MCEM | 15.747 | 14.422 | −1.33 |
| Burr+INLA† | 13.925 | 11.924 | −2.00 |

> † Burr+INLA uses approximate hardcoded parameters; fully optimised rank-ASSE ≈ 12.78 (see Comprehensive Scoreboard above).  
> Small discrepancies vs `rfl_compare_all.py` (e.g. Normal+INLA: 12.872 vs 12.850) reflect parameter rounding.

**Key finding:** z-ASSE consistently improves over rank-ASSE for continuous-prior methods. The improvement is largest for SEV-based models — the asymmetric SEV error distribution creates a systematic offset between rank-matched marginal quantiles and within-group z-score positions. For the complete picture including all methods and both metrics, see the Comprehensive Scoreboard above.


## 7. Direct ASSE Optimisation via Heuristic Learning (`rfl_chiu.py`)

**File:** `rfl_chiu.py`  
**Goal:** Reproduce the Chiu (2005) thesis prediction, understand its ASSE criterion precisely, and beat the thesis benchmark via direct optimisation.

### 7.1 Four Methods

| Label | Description | z-ASSE | rank-ASSE | Speed |
|-------|-------------|-------:|----------:|-------|
| **A — Chiu EIV (hard-coded)** | Exact thesis parameters (Table 3) | 10.78 | 12.41 | instant |
| **B — Direct rank-ASSE opt** | **3-stage heuristic** (grid→SA→NM) over 5 params, 40-pt GH | 11.23 | **12.24** ← new best | ~60–90 s |
| **C-1 — OLS z-ASSE opt** | Optimise $(\mu_\Delta, \sigma_\Delta)$; OLS $(\beta_0, \beta_1)$ | **10.31** | — | <0.1 s |
| **C-2 — LAD z-ASSE opt** | Optimise all 4 params; LAD $(\beta_0, \beta_1)$ | **9.94** 🏆 | — | <0.1 s |

### 7.2 Two ASSE Metrics — Not Interchangeable

**z-ASSE** (Chiu 2005 original):
1. Compute within-group sample z-score: $z_t = (y_t - \bar y_j) / s_j$
2. Map to fatigue-limit quantile: $\hat\Delta_t = \exp(\mu_\Delta + \sigma_\Delta z_t)$
3. Compute predicted log-life: $\hat\omega_t = \beta_0 + \beta_1 \ln(S_t - \hat\Delta_t)$
4. Fit OLS regression $\omega_t \sim q_t = \ln(S_t - \hat\Delta_t)$ to get $(\hat\beta_0, \hat\beta_1)$
5. ASSE$_z = \sum_t |\omega_t - \hat\omega_t|$

**rank-ASSE** (P&M 1999 Response, the standard E criterion):
$$E = \sum_j \sum_i \left|\ln y_{(i)j} - F_W^{-1}\!\!\left(\frac{i-0.5}{n_j};\; s_j,\; \hat\theta\right)\right|$$
where $F_W^{-1}$ is the marginal CDF inverted numerically (vectorised grid + interp, ~50× faster than brentq).

> Chiu's thesis value of **10.80 is z-ASSE**. Most other entries in this repo are **rank-ASSE**. Do not compare directly.

### 7.3 Why C-2 LAD Beats OLS (Chiu's Approach)

Chiu's EIV uses OLS to fit $(\beta_0, \beta_1)$ — minimising the sum of *squared* prediction errors. But the ASSE metric is the sum of *absolute* prediction errors (SAE). The correct minimiser is **LAD regression** (least absolute deviations), which minimises SAE directly:

$$\min_{\beta_0,\beta_1}\; \sum_t |\omega_t - \beta_0 - \beta_1 q_t|$$

This is solved as a linear program (scipy linprog interior-point). Combined with simultaneously optimising $(\mu_\Delta, \sigma_\Delta)$ via Nelder-Mead over the LAD objective, C-2 achieves z-ASSE = **9.94** — an 8% improvement over the thesis.

### 7.4 Censoring Support via KM-Based z-Scores

When data include right-censored observations (Type I, Type II, hybrid, or progressive), the within-group z-scores $z_t = (y_t - \bar y_j)/s_j$ are biased (censored observations are excluded from $\bar y_j$ and $s_j$).

`rfl_chiu.py` provides `build_km_z()` which replaces within-group z-scores with **Kaplan–Meier plotting positions**:

$$z_t = \Phi^{-1}(p_t^{\text{KM}}), \qquad p_t^{\text{KM}} = (1 - \hat S_{\text{KM}}(t)) - \frac{0.5}{n}$$

The Hazen correction $-0.5/n$ ensures consistency with the $(i-0.5)/n$ plotting positions used in rank-ASSE, and caps $p_t^{\text{KM}} \in (10^{-6},\, 1-10^{-6})$ to avoid $\pm\infty$ z-scores.

This gives a unified interface: any censoring schema is reduced to a set of $(S_j, \omega_t, z_t)$ triples, and all four methods (A, B, C-1, C-2) work identically on these triples.

**Censoring types handled:**

| Type | Description | Implementation |
|------|-------------|---------------|
| None | All observations are failures | Standard rank ordering |
| Type I | Fixed censoring time $T_j$ per group | KM on $(t, \delta)$ pairs |
| Type II | Censor after $r$-th failure (fixed $r$, random $T$) | Same KM interface (slight non-independence caveat) |
| Hybrid I-II | $T^* = \min(y_{(r)}, T)$ | Same KM interface |
| Progressive | Different censoring times per observation | Same KM interface |

### 7.5 Heuristic Learning in Method B (`run_rank_asse_opt()`)

The rank-ASSE landscape over $\theta = (\beta_0, \beta_1, \log\sigma_\varepsilon, \mu_\Delta, \log\sigma_\Delta)$ is **non-convex for the same structural reason as the log-likelihood** — the $\log(S - \Delta)$ nonlinearity creates multiple local minima in a 5D space. `run_rank_asse_opt()` therefore applies the same 3-stage heuristic pipeline as `heuristic_optimize()` in `rfl_inla.py`:

```
Stage 1  Random grid (30 pts, always includes Chiu warm-start)  → coarse ASSE survey
Stage 2  Dual Annealing (500 iters, embedded L-BFGS-B)          → adaptive global escape
Stage 3  Nelder-Mead polish (xatol/fatol=1e-7)                  → sub-grid convergence
```

This is methodologically consistent: **heuristic learning is the cross-cutting optimisation strategy** for all non-convex objectives in this repo — whether the objective is the log-likelihood (INLA methods) or the rank-ASSE directly (Method B).

Methods C-1/C-2 use simpler Nelder-Mead because their effective parameter space is 2D (outer $\mu_\Delta, \sigma_\Delta$; inner $\beta_0, \beta_1$ solved analytically/by LP) and the z-ASSE landscape is cheaper to evaluate — the single-basin structure at those scales makes multi-stage search unnecessary.

### 7.6 CRPS Alternative (`rfl_crps.py`)

`rfl_crps.py` implements the **marginal CRPS** as an alternative optimisation target, using the energy-score decomposition:

$$\text{CRPS}(F_W, y) = \mathbb{E}_\Delta[|X-y|] - \frac{1}{2}\mathbb{E}_{\Delta,\Delta'}[|X-X'|]$$

where both terms are computed via 40-pt Gauss–Hermite over the LogNormal prior. The Jensen gap (averaging conditional CRPSes) is avoided by computing the full marginal energy score.

**Finding:** CRPS optimisation converges to $\sigma_\varepsilon \approx 0.29$, which gives rank-ASSE = 12.65 — slightly worse than direct rank-ASSE optimisation (12.24). CRPS and rank-ASSE have **different optima**; CRPS is not a reliable proxy for rank-ASSE minimisation.

### 7.7 Evolution of Methods

```
Chiu (2005) thesis      — Error-in-variables regression, ASSE = 10.80
          ↓
rfl_em.py               — EM + NPMLE, BIC for K selection
          ↓
rfl_profile.py          — Normal + NPMLE + Profile Likelihood SE (7.7x Louis correction)
          ↓
rfl_sev.py              — SEV + NPMLE: Weibull conditional (weakest-link physics)
                          AIC/BIC beat Normal+NPMLE; alpha=-b1/sig=15.0, Weibull shape=2.35
          ↓
rfl_mcem.py             — SEV + MCEM: continuous LogNormal g(Delta), rejection sampling E-step
                          ASSE = 10.89 (≈ Chiu thesis), beats all discrete NPMLE variants
          ↓
rfl_inla.py             — Normal + INLA: 9-pt Gauss-Hermite + dual-annealing, per-obs residual = 8.63
          ↓
rfl_sev_inla.py ⭐      — SEV + INLA: SEV conditional + 9-pt GH + dual-annealing
                          per-obs residual = 5.76 (best in-sample, -33% vs Normal+INLA)
          ↓
ssla_se.py              — SSLA-based SE: deterministic fix for Louis 7.7x underestimation + full 5D UQ
          ↓
rfl_burr.py             — Burr XII MLE v1 (4-param): Gamma conjugate closed form
                          b1/S_i unidentifiable; MLE degenerates b->0; ASSE=3.59 artifactual
          ↓
rfl_burr2.py            — Burr XII MLE v2 (5-param): stress-dependent prior b_Si=a/delta_i^alpha
                          All 5 params identifiable; ASSE=15.92 (underperforms SEV+INLA)
                          log-lik = -80.11 vs SEV+INLA -72.88 (genuine model misspecification)
          ↓
rfl_burr_inla.py        — Burr XII + INLA (6-param): Burr XII as conditional f(y|Delta_i),
                          INLA integrates Delta_i; nests SEV+INLA (a->inf)
                          a=23.2, per-obs residual=5.74, log-lik=-72.884 (AIC=157.77 vs SEV+INLA 155.76)
                          Conclusion: SEV+INLA (5 params) still preferred by AIC
          ↓
rfl_burr_em.py          — Burr XII + EM-GMM: K-component Gaussian mixture prior on Delta_i,
                          trapezoidal grid E-step (N=400 pts), soft assignments, L-BFGS-B M-step
                          Mode A (sig>=0.15 constraint, K=1): sig=0.160, a=1.6, ASSE=4.09 (best)
                          AIC=157.73 vs SEV+INLA 155.76 (SEV+INLA preferred by AIC)
```

## 8. Standard Error Estimation (`ssla_se.py`)

The `se_hessian()` in `rfl_inla.py` holds $(\mu_\Delta, \sigma_\Delta)$ fixed, producing a 3×3 Hessian and inheriting the same Louis-style underestimation problem that affects `rfl_em.py`. `ssla_se.py` provides a **deterministic, sampling-free** alternative based on the Self-Supervised Laplace Approximation (Rodemann et al., TMLR 2026, arXiv:2605.12208).

### 8.1 Core Idea

Rather than approximating the parameter posterior $p(\theta|\mathcal{D})$, SSLA directly quantifies uncertainty by **refitting on self-predicted data**:

$$\text{SE}(\hat\theta) \approx |\tilde\theta - \hat\theta|$$

where $\tilde\theta$ is the refit on $\hat{Y} = \hat\beta_0 + \hat\beta_1 \log(S_i - \hat\Delta_i)$ (the model's own predictions). The sensitivity of the fitted parameters to the data is the uncertainty estimate.

### 8.2 Two Directions

#### Direction A — ASSLA-EM (fix Louis 7.7× underestimation)

```python
from ssla_se import assla_se_em, em_rfl

# Get EM fit
res_em = em_rfl(Y, S, K=2, delta_init=[0.52, 0.57])

# ASSLA SE — no Bootstrap needed
se = assla_se_em(Y, S, res_em)
print(f"SE(b0) = {se['se_b0']:.4f}")
print(f"SE(b1) = {se['se_b1']:.4f}")   # Louis gives ~0.193; Profile SE ~1.486
print(f"SE(sig)= {se['se_sig']:.4f}")
```

Algorithm:
1. MAP fatigue limit per obs: $\hat\Delta_i = \Delta_{\arg\max_k \tau_{ik}}$
2. Self-predict: $\hat Y_i = \hat\beta_0 + \hat\beta_1 \log(S_i - \hat\Delta_i)$
3. Refit EM on $\hat Y$ (4 restarts, same $K$)
4. $\text{SE}(\hat\theta) = |\tilde\theta - \hat\theta|$

#### Direction B — SSLA-INLA (full 5D uncertainty quantification)

Replaces the nuisance-fixed 3×3 `se_hessian()` with a full 5-parameter sensitivity estimate that includes $(\mu_\Delta, \sigma_\Delta)$:

```python
from ssla_se import ssla_se_inla, assla_se_inla_norefit, heuristic_optimize
import numpy as np

cens = np.zeros(n, bool)
theta_hat, ll_hat = heuristic_optimize(Y, S, cens)

# Fast variant: full 5D Hessian (no refit)
se_fast = assla_se_inla_norefit(Y, S, theta_hat)
print(f"SE(b1) = {se_fast['se_b1']:.4f}  SE(mu_d) = {se_fast['se_mu_d']:.4f}")

# Full SSLA: refit on self-predictions (more faithful, ~2× slower)
se_full = ssla_se_inla(Y, S, theta_hat, n_grid=10, sa_maxiter=200)
print(f"SE(b1) = {se_full['se_b1']:.4f}  SE(sig_d) = {se_full['se_sig_d']:.4f}")
```

### 8.3 SE Method Comparison

| Method | SE(β₀) | SE(β₁) | SE(σ) | Notes |
|--------|--------|--------|-------|-------|
| Profile SE (rfl_profile.py) | 0.580 | **1.486** | 0.054 | Gold standard, profile likelihood |
| Louis SE (rfl_em.py) | 0.360 | **0.193** | 0.052 | 7.7× underestimate of SE(β₁) |
| ASSLA-EM (`ssla_se.py`) | 0.204 | **0.589** | ~~0.563~~ ⚠️ | SE(σ) unreliable: σ collapses on noiseless refit |
| ASSLA-INLA norefit (`ssla_se.py`) | 0.436 | **1.050** | 0.153 | Full 5D Hessian; SE(β₁) 4.3× > 3×3 Hessian |
| SSLA-INLA refit (`ssla_se.py`) | 0.117 | **0.353** | ~~0.285~~ ⚠️ | Refit on noiseless Y_self degrades estimates |

**Actual results on n=75 data** (see `_test_ssla.py`):

| Method | SE(β₀) | SE(β₁) | SE(σ) |
|--------|--------|--------|-------|
| Profile SE | 0.580 | 1.486 | 0.054 |
| Louis SE (EM) | 0.360 | 0.193 | 0.052 |
| ASSLA-EM | 0.204 | **0.589** | ~~0.563~~ ⚠️ |
| ASSLA-INLA 5D Hessian | 0.436 | **1.052** | 0.152 |

> **Winner: ASSLA-INLA norefit** — SE(β₁) = 1.050, closest to Profile 1.486 (gap reflects model difference: LogNormal vs NPMLE).  
> ⚠️ SE(σ) is unreliable in both ASSLA-EM and SSLA-INLA refit: self-predicted Y_self has zero residuals, σ collapses on refit.  
> ⚠️ SSLA-INLA refit is **worse** than norefit for β₁ (0.353 < 1.050): refitting on noiseless data degrades estimates. The ASSLA approximation (no refit) is the recommended approach.

### 8.4 Why SSLA Instead of Bootstrap?

| Criterion | Bootstrap ($R=500$) | SSLA/ASSLA |
|-----------|--------------------|-----------:|
| Computation | $500 \times$ EM time | $1$–$4 \times$ EM time |
| Randomness | Stochastic (varies by seed) | Deterministic |
| Full 5D SE | Yes (via B resamples) | Yes (via Hessian / refit) |
| Implementation | External loop | Single function call |

Reference: Rodemann J., Marquard A., Augustin T., Caprio M. (2026). *Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification*. TMLR. arXiv:2605.12208.

## 9. Burr XII Closed-Form Marginal MLE (`rfl_burr.py`, `rfl_burr2.py`)

These two scripts explore whether **the Gamma-conjugate closed form** can replace numerical integration entirely, yielding a fully closed-form MLE without any quadrature.

### 9.1 Theoretical Background

The SEV density rewrites as:

$$f(\ln Y_i \mid \Delta_i) = \frac{c_i}{\sigma} \cdot V_i \cdot e^{-c_i V_i}, \qquad V_i = (S_i - \Delta_i)^{-\beta_1/\sigma},\quad c_i = e^{(\ln Y_i - \beta_0)/\sigma}$$

This is a **Gamma(2, $c_i$) kernel** in $V_i$. Placing a Gamma$(\alpha_0, b)$ prior on $V_i$ and integrating out gives the **Burr Type XII** marginal:

$$L_i = \frac{\alpha_0 \, c_i \, b^{\alpha_0}}{\sigma\,(c_i + b)^{\alpha_0 + 1}}$$

No numerical integration — fully closed form.

### 9.2 Version 1: `rfl_burr.py` — 4-Parameter Model (Degenerate)

**Problem**: the rate $b$ is a single constant shared across all observations. Once $V_i$ is integrated out with a fixed-rate Gamma prior, $\beta_1$ and $S_i$ are **absorbed into $b$** and become unidentifiable. The 4-parameter model $(b_0, \sigma, \alpha_0, b)$ has no access to stress-level structure.

**Consequence**: the MLE drives $b \to 0$, making the posterior mean of $V_i$ degenerate and the fitted value $\hat{y}_i \approx y_i - \text{const}$. ASSE = 3.59 is artifactual — it's a trivial memorisation solution, not a predictive model.

**The fix required**: restore $\beta_1$ and $S_i$ by using a **stress-dependent prior rate**.

### 9.3 Version 2: `rfl_burr2.py` — 5-Parameter Model (Stress-Dependent Prior)

**Key idea**: set the prior rate to $b_{S_i} = a / \delta_i^\alpha$ where $\delta_i = S_i - \mu_\Delta$ and $\alpha = -\beta_1/\sigma$. This centres the Gamma prior at $E[V_i] = \delta_i^\alpha = (S_i - \mu_\Delta)^\alpha$ — the expected value of $V_i$ at the prior mean fatigue limit.

This substitution yields a clean closed-form marginal with **all 5 parameters appearing**:

$$\boxed{L_i = \frac{a^{a+1}}{\sigma} \cdot \frac{e^{w_i}}{(a + e^{w_i})^{a+1}}}$$

where:
$$w_i = \frac{\ln Y_i - \mu_i^*}{\sigma}, \qquad \mu_i^* = \beta_0 + \beta_1 \log(S_i - \mu_\Delta)$$

The 5 score equations at the MLE (letting $p_i = e^{w_i}/(a + e^{w_i})$, $r_i = 1 - (a+1)p_i$):

| Parameter | Score condition |
|-----------|----------------|
| $\beta_0$ | $\bar{r} = 0 \;\Leftrightarrow\; \bar{p} = 1/(a+1)$ |
| $\beta_1$ | $\sum r_i \log \delta_i = 0$ |
| $\sigma$ | $\sum r_i w_i = -n$ |
| $\mu_\Delta$ | $\sum r_i / \delta_i = 0$ |
| $a$ | $\log a + 1/a = \overline{\log(a + e^{w_i})}$ |

**Fitted value** (posterior predictive mean of $Y_i$):

$$\hat{y}_i = \mu_i^* - \sigma\bigl(\psi(a+1) - \log(a + e^{w_i}) + \gamma_E\bigr)$$

### 9.3.1 MLE Results (`rfl_burr2.py` on P&M 1999 Data)

```
b0     = -9.340   b1     = -8.321
sigma  =  0.327   a      =  0.638
mu_d   =  0.528   log-lik = -80.11

ASSE = 15.92   MAE = 0.212
```

Score checks: all 5 equations satisfied to $<10^{-9}$ — the MLE was found correctly.

### 9.3.2 Why Burr XII MLE Underperforms SEV+INLA

| | SEV+INLA | Burr XII v2 |
|---|---|---|
| Model for $\Delta_i$ | LogNormal$(\mu_d, \sigma_d^2)$ — individual distribution | All $\Delta_i$ share single point $\mu_\Delta$, variability in $V_i$ | 
| Parameters | 5: $\beta_0, \beta_1, \sigma, \mu_d, \sigma_d$ | 5: $\beta_0, \beta_1, \sigma, a, \mu_\Delta$ |
| log-lik | **−72.88** | −80.11 |
| Individual $\Delta_i$ posterior | Per-observation GH integral over LogNormal | Shared via $a$ shape parameter |
| ASSE(n=75) rank-matched | **13.02** | — (not computed) |

The closed-form gain (no quadrature) comes at the cost of losing **between-individual heterogeneity** in $\Delta_i$: the stress-dependent prior centres all observations at the same $\mu_\Delta$, with the Gamma shape $a$ capturing only collective variability. SEV+INLA's $\sigma_d$ (LogNormal scale) is a genuine per-specimen variance term that SEV+INLA can integrate over — Burr XII has no equivalent.

**Conclusion**: the Burr XII closed-form marginal MLE is theoretically clean and computationally fast (no inner loop), but it is a structurally weaker model than SEV+INLA. The 7.23-unit log-likelihood gap confirms a genuine model misspecification, not just an approximation error.

### 9.4 Version 3: `rfl_burr_inla.py` — 3-Level Hierarchy + INLA

**Key idea** (Roy's suggestion): instead of using the Burr XII as the *complete* marginal (Δ already integrated out), use it as the *conditional* likelihood given Δ_i, and let INLA integrate Δ_i over its LogNormal prior.

**3-level hierarchy**:
1. $\Delta_i \sim \text{LogNormal}(\mu_d, \sigma_d^2)$ — INLA outer integral
2. $U_i | \Delta_i \sim \text{Gamma}(a,\; a/(S_i-\Delta_i)^\alpha)$ — Gamma conjugate, closed form
3. $Y_i | U_i \sim \text{SEV}(b_0 - \sigma\log U_i,\; \sigma)$ — conditional SEV

Marginalizing (2)+(3) gives the Burr XII conditional density:
$$f_\text{Burr}(\ln Y_i \mid \Delta_i) = \frac{a^{a+1}}{\sigma}\cdot\frac{e^{w_i(\Delta_i)}}{(a + e^{w_i(\Delta_i)})^{a+1}}$$

INLA then integrates over $\Delta_i$. The model has 6 parameters: $(\beta_0, \beta_1, \sigma, a, \mu_d, \sigma_d)$.

**Critical mathematical property**: as $a \to \infty$, $f_\text{Burr}(\ln Y_i | \Delta_i) \to f_\text{SEV}(\ln Y_i | \Delta_i)$, so this model **nests `rfl_sev_inla.py` exactly**.

**Results**:

```
b0=-9.316  b1=-8.522  sig=0.189  a=23.20  mu_d=-0.644  sig_d=0.035
log-lik = -72.884   (SEV+INLA: -72.880)
per-obs residual =   5.74    (SEV+INLA:   5.76)
```

**The a=23.2 finding**: in `rfl_burr2.py` (no per-obs Δ_i), the MLE drove $a=0.638$ — very heavy-tailed Gamma to compensate for the missing individual heterogeneity. Once INLA restores per-observation Δ_i posteriors, only mild overdispersion is needed ($a=23.2$). The two sources of variability are **complementary**, not additive.

**AIC comparison**:

| Model | log-lik | params | AIC |
|-------|:-------:|:------:|:---:|
| SEV+INLA (`rfl_sev_inla.py`) | −72.880 | 5 | **155.76** |
| Burr XII+INLA (`rfl_burr_inla.py`) | −72.884 | 6 | 157.77 |

AIC slightly favours SEV+INLA. The 0.004-nats log-likelihood gain from the extra $a$ parameter is negligible. **`rfl_sev_inla.py` remains the recommended method** (per-obs residual=5.76, AIC=155.76, 5 params).

### 9.5 Version 4: `rfl_burr_em.py` — Burr XII + EM-GMM Prior

**Key idea**: Instead of a parametric LogNormal prior on $\Delta_i$ (SEV+INLA / Burr+INLA), use a **$K$-component Gaussian mixture** and learn it jointly with the likelihood parameters via EM. This bridges the NPMLE approach (data-driven discrete mixing) with the Burr XII closed-form inner likelihood.

**4-level generative model**:

1. $\Delta_i \sim \sum_{k=1}^K \pi_k\, \mathcal{N}(\mu_k, \sigma_k^2)$ — GMM prior on fatigue limit, EM-estimated
2. $U_i \mid \Delta_i \sim \mathrm{Gamma}\!\bigl(a,\; a/(S_i-\Delta_i)^\alpha\bigr)$ — Gamma conjugate (closed form)
3. $Y_i \mid U_i \sim \mathrm{SEV}(b_0 - \sigma \log U_i,\; \sigma)$ — conditional SEV

Marginalising out $U_i$ in layers (2)+(3) gives the **Burr XII conditional likelihood**:

$$f_\text{Burr}(\ln Y_i \mid \Delta_i) = \frac{a^{a+1}}{\sigma}\cdot\frac{e^{w_i(\Delta_i)}}{\bigl(a + e^{w_i(\Delta_i)}\bigr)^{a+1}}, \quad w_i(\Delta_i) = \frac{\ln Y_i - b_0 - b_1\log(S_i-\Delta_i)}{\sigma}$$

The remaining integral over $\Delta_i$ is handled by a **trapezoidal grid** (400 uniform points on $\Delta \in [0.002, 0.670]$), not Gauss–Hermite.

#### EM Algorithm

**E-step** — for each observation $i$ and GMM component $k$:

$$\log L_{ik} = \log \int f_\text{Burr}(\ln Y_i \mid \delta)\, \mathcal{N}(\delta;\mu_k,\sigma_k^2)\, d\delta \quad \text{(trapezoidal)}$$

$$r_{ik} = \frac{\pi_k\, L_{ik}}{\sum_{k'} \pi_{k'} L_{ik'}}, \quad E[\Delta_i \mid Y_i, Z_i=k] = \frac{\sum_j w_j^{(k)} \delta_j}{\sum_j w_j^{(k)}}$$

where $w_j^{(k)} \propto f_\text{Burr}(\ln Y_i \mid \delta_j)\, \mathcal{N}(\delta_j;\mu_k,\sigma_k^2)$ are the per-grid-point weights.

**M-step GMM** — closed form given soft assignments $r_{ik}$ and posterior moments:

$$\pi_k^{\text{new}} = \frac{1}{n}\sum_i r_{ik}, \qquad \mu_k^{\text{new}} = \frac{\sum_i r_{ik}\, E[\Delta_i \mid k]}{\sum_i r_{ik}}, \qquad \sigma_k^{2,\text{new}} = \frac{\sum_i r_{ik}\, E[\Delta_i^2 \mid k]}{\sum_i r_{ik}} - (\mu_k^{\text{new}})^2$$

**M-step likelihood** — L-BFGS-B on the mixture marginal log-likelihood:

$$\ell(b_0, b_1, \sigma, a) = \sum_i \log\sum_k \pi_k L_{ik}$$

#### The Degenerate Solution Problem

Without constraints, the unconstrained MLE drives $\sigma \to 0$ (lower bound 0.05) and $a \to \infty$ (upper bound 200), giving ASSE = 0.49 — apparent **overfitting via degenerate Burr XII**:

As $\sigma \to 0$ and $a \to \infty$, the Burr XII density $f_\text{Burr}(\ln Y_i \mid \delta)$ concentrates near a single point $\delta_i^* = S_i - e^{(\ln Y_i - b_0)/b_1}$ (the implied fatigue limit for each observation). The GMM prior then memorises all $\delta_i^*$ values, making $\widehat{\ln Y_i} \approx \ln Y_i$. Unlike Gauss–Hermite (which has curvature bias away from the mode), the trapezoidal grid accurately captures this sharp peak — enabling the degenerate solution to be found.

This is a genuine identifiability issue: the Burr XII + GMM model without regularisation can always memorise all observations.

#### Three Modes

Three constrained configurations were tested to escape the degenerate solution:

| Mode | Constraints | sig | a | ASSE | AIC |
|------|-------------|:---:|:-:|:----:|:---:|
| Unconstrained | sig ≥ 0.05, a ≤ 200 | 0.05† | 200† | 0.49† | — |
| **Mode A** | sig ≥ 0.15, a ≤ 50 | **0.160** | **1.6** | **4.09** | **157.73** |
| Mode B | sig fixed at SEV+INLA (0.190) | 0.190 | ≈9.5 | ≈5.6 | ≈159.8 |
| Mode C | sig=0.190 and a=23.2 fixed | 0.190 | 23.2 | ≈5.7 | — |

† Degenerate; ASSE not meaningful.

**Mode A K=1 is the best result** (ASSE=4.09), finding a parameter configuration that neither SEV+INLA nor Burr+INLA's Laplace approximation can access: $a=1.6$ (much lighter-tailed Gamma prior on $V_i$ than Burr+INLA's $a=23.2$) combined with $\sigma=0.160$ (tighter than SEV+INLA's 0.190). The Laplace approximation has curvature bias that steers it away from this region.

**AIC comparison with prior methods**:

| Model | log-lik | params | AIC | per-obs residual |
|-------|:-------:|:------:|:---:|-----------------:|
| SEV+INLA (`rfl_sev_inla.py`) | −72.880 | 5 | **155.76** | 5.76 |
| Burr XII+INLA (`rfl_burr_inla.py`) | −72.884 | 6 | 157.77 | 5.74 |
| Burr XII+EM-GMM Mode A K=1 (`rfl_burr_em.py`) | ≈−72.87 | 6 | 157.73 | **4.09** |

**Interpretation**: Mode A achieves the best ASSE by finding a sharper per-observation posterior for $\Delta_i$ (small $\sigma = 0.160$, very mild Gamma prior $a=1.6$), but the AIC still slightly favours SEV+INLA. The $\sigma \geq 0.15$ constraint is heuristic — a formal regularisation framework (e.g., penalised likelihood) would be needed to justify this bound from first principles.

## 10. Conclusions

### 10.1 Summary of Results

All methods evaluated on the Pascual & Meeker (1999) fatigue dataset ($n=75$, five stress levels, 15 per level). Two metrics are reported side-by-side — **rank-ASSE** (P&M 1999, marginal CDF calibration) and **z-ASSE** (Chiu 2005, within-group ordering). They are independent metrics on different scales; comparison is only valid within the same column. Lower = better.

| Method | Source | Params | rank-ASSE | z-ASSE |
|--------|--------|:------:|----------:|-------:|
| **Method C-2 — LAD z-opt** (`rfl_chiu.py`) 🏆 | This work | 4 | — | **9.94** |
| Method C-1 — OLS z-opt (`rfl_chiu.py`) | This work | 4 | — | 10.31 |
| **Method B — direct rank-ASSE opt** (`rfl_chiu.py`) ⭐ | This work | 5 | **12.24** | — |
| Chiu (2005) EIV | Literature | 5 | 12.41\* | 10.80 |
| Burr XII + INLA (`rfl_burr_inla.py`) | This work | 6 | 12.78 | 11.92† |
| Pascual & Meeker (1999), Nor–Nor | Literature | 5 | 12.84 | — |
| Burr XII + EM-GMM (`rfl_burr_em.py`) | This work | 6 | 12.84 | 12.76‡ |
| Normal + INLA (`rfl_inla.py`) | This work | 5 | 12.85 | 12.75 |
| SEV + INLA (`rfl_sev_inla.py`) | This work | 5 | 13.02 | 11.50 |
| SEV + MCEM (`rfl_mcem.py`) | This work | 5 | 15.75§ | 14.42 |
| SEV + NPMLE (`rfl_sev.py`) | This work | 6 | 16.49 | — |
| Normal + NPMLE (`rfl_profile.py`) | This work | 6 | 16.85 | — |
| Spindel & Haibach (1981) | Literature | 6 | 17.35 | — |
| Castillo & Hadi (1995) | Literature | 5 | 18.12 | — |
| Castillo et al. (1985) | Literature | 4 | 20.27 | — |
| Bastenaire (1972) | Literature | 5 | 20.52 | — |
| Little & Ekvall (1981), model 2 | Literature | 3 | 31.17 | — |
| Little & Ekvall (1981), model 1 | Literature | 3 | 41.13 | — |

> \* Chiu (2005) **rank-ASSE = 12.41** computed by this work using thesis parameters (Method A, `rfl_chiu.py`). The thesis value **10.80 is z-ASSE** — a different metric; do not compare to rank-ASSE column.  
> † Burr+INLA z-ASSE uses approximate hardcoded parameters (`rfl_asse_zscore.py`).  
> ‡ Burr+EM-GMM z-ASSE uses Normal GMM prior quantile $\hat\delta_{ij}=\mu_1+\sigma_1 z_{ij}$ ($\mu_1=0.526$, $\sigma_1=0.0185$); approximation only.  
> § SEV+MCEM from hardcoded warm-start; fully converged MLE may differ.  
> "—" in rank-ASSE for C-1/C-2: these methods optimise z-ASSE directly; rank-ASSE not separately computed.

**Censoring robustness** (§5.4, three scenarios [A]=0%, [B]=20%, [C]=37.3%):

| Method | [A] Fitted MAE | [B] Fitted MAE | [C] Fitted MAE | [C] Status |
|--------|:--------------:|:--------------:|:--------------:|:----------:|
| Normal + INLA | 8.63 | n/a | n/a | ❌ $\hat\beta_1\to-20$ |
| SEV + INLA | 5.76 | 5.29 | 17.97 | ❌ $\hat\beta_1\to-20$ |
| **Burr+EM-GMM** ($a\geq1$) | **4.09** | **4.92** | **4.43** | **✅ stable** |

**Key findings:**

1. **rank-ASSE champion**: Method B (direct optimisation, `rfl_chiu.py`) = **12.24** — beats Chiu EIV (12.41) and all Bayesian integration methods.
2. **z-ASSE champion**: Method C-2 LAD (`rfl_chiu.py`) = **9.94** — beats Chiu's thesis (10.80) by **8%**, by using LAD instead of OLS to directly minimise the SAE metric.
3. **Chiu's 10.80 is z-ASSE, not rank-ASSE.** On rank-ASSE, Chiu EIV = 12.41; our Method B = 12.24 is better.
4. **MLE ≠ ASSE minimisation**: Normal+INLA MLE achieves rank-ASSE = 12.85; Method B direct optimisation achieves 12.24 by choosing $\hat\sigma_\varepsilon = 0.258$ vs MLE's 0.131.
5. **INLA-class** methods cluster at rank-ASSE ≈ 12.8–13.0, matching P&M's benchmark (12.84). Normal+INLA = 12.85 ✓.
6. **NPMLE** (discrete prior K=2) is systematically worse (16.5–16.9) — coarse discrete $g(\Delta)$ distorts the marginal CDF shape.
7. **Censoring**: only Burr+EM-GMM remains stable at 37.3% censoring; GH-based methods collapse.

### 10.2 Theoretical Comparison of Methods

> **Note**: the theoretical discussion below refers to model comparison on the ASSE(n=75) rank-matched criterion. Previous versions of this section incorrectly computed "ASSE" as $\sum|y_i - E[\mu(S_i,\Delta_i)|y_i]|$ (posterior residuals conditioned on $y_i$). That quantity is not ASSE — it is an in-sample posterior residual that decreases as the posterior on $\Delta_i$ concentrates. It cannot be compared across model classes or against the thesis values.

#### 1. Discrete vs Continuous Integration over $\Delta$

- **NPMLE** (K=2 atoms): the discrete prior concentrates mass at only 2 support points $\{\hat\Delta_1, \hat\Delta_2\}$. The marginal CDF $F_W(y; s) = \sum_k \pi_k F_\text{cond}(y; s, \Delta_k)$ is a mixture of only 2 components, giving ASSE(n=75) $\approx 16$–17.

- **INLA / EM-GMM**: continuous LogNormal or GMM prior allows the marginal CDF to have a smooth shape that better matches the empirical distribution within each stress group. These methods achieve ASSE(n=75) $\approx 12.8$–13.0, matching P&M's benchmark.

The key result: $E_{\text{discrete-}K} \gg E_{\text{continuous}}$ for small $K$, regardless of Normal or SEV conditional.

#### 2. GH Quadrature vs Monte Carlo E-step

Both MCEM and SEV+INLA use a continuous LogNormal $g(\Delta)$, but differ in how they approximate the marginal likelihood integral:

| Criterion | MCEM (M=200) | 9-pt GH (INLA) |
|-----------|:---:|:---:|
| Deterministic | No (MC noise ±1–2 in LL) | Yes |
| Iterations to converge | 70–80 | 1 (direct MLE) |
| ASSE(n=75) — SEV variant | — | 13.02 |
| LL at convergence | −12.5 (MC approx) | −72.88 (exact) |

The 9-pt GH evaluates the integrand at optimal quadrature nodes centred at the per-observation posterior mode, achieving far higher accuracy than 200 random samples. The MC noise in MCEM also pollutes the M-step BFGS, preventing convergence to the true MLE; this partly explains the 1.88 log-unit gap between MCEM ($\ell \approx -74.76$) and SEV+INLA ($\ell = -72.88$).

### 10.3 Pros and Cons

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **NPMLE + Profile SE** (`rfl_profile.py`) | Gold-standard SE calibration; corrects Louis 7.7× underestimate; no parametric $g$ assumption | High ASSE; discrete $\Delta$ too coarse for prediction |
| **SEV + NPMLE** (`rfl_sev.py`) | Physical motivation (weakest-link); better AIC/BIC than Normal+NPMLE | Same ASSE limitation as Normal+NPMLE |
| **SEV + MCEM** (`rfl_mcem.py`) | Continuous $g(\Delta)$; good warm-start for INLA; flexible | MC noise; slow convergence; needs M≥200, iter≥70 |
| **Normal + INLA** (`rfl_inla.py`) | Accurate GH integration; fast NM/SA convergence | Symmetric Normal misspecifies left-skewed log-life; ASSE 8.63 |
| **SEV + INLA** (`rfl_sev_inla.py`) | Best per-obs residual (5.76) among GH methods; physically correct conditional; deterministic | Degenerates at 37.3% censoring ($\hat\beta_1\to-20$, same as Normal+INLA); SE not yet Profile-calibrated |
| **Burr XII + EM-GMM** (`rfl_burr_em.py`) | Best per-obs residual (4.09); **robust under heavy censoring** ([C] per-obs residual=4.43 vs SEV+INLA 17.97); no Laplace curvature bias | Degenerate MLE without $\sigma,a$ constraints; heuristic bounds; AIC still favours SEV+INLA |

### 10.4 Limitations

1. **In-sample ASSE (addressed by Direction A):** LOO analysis confirms all ASSE figures reflect posterior reconstruction quality, not new-specimen prediction. Prior-LOO ASSE ≈ 40 for all methods — the S-level marginal mean is the effective prediction baseline.

2. **Small sample ($n=75$)**: with only 15 specimens per stress level, parameter estimates and ASSE rankings may not generalise to other fatigue datasets. The near-degenerate $g(\Delta)$ ($\hat\sigma_\Delta \approx 0.03$) in particular may be an artefact of small $n$.

3. **Unimodal LogNormal $g(\Delta)$ (partially addressed by Direction C):** K=3 LogNormal mixture improves AIC by 8.2 units but BIC still favours K=1. The K=3 improvement degenerates to ≈2 distinct modes — genuine multimodality cannot be confirmed at $n=75$.

4. **SE for SEV+INLA not Profile-calibrated (addressed by Direction B):** Laplace approximation posterior SEs from $H_\text{obs}^{-1}$: $\beta_0$ ±0.30, $\beta_1$ ±0.36, $\sigma$ ±0.045. Profile likelihood calibration (as in `rfl_profile.py`) remains to be implemented for the 5D SEV+INLA parametrisation.

5. **Censoring robustness**: simulated censoring experiments (see Cross-Method Censoring Comparison above) show that GH-based methods (Normal+INLA, SEV+INLA) both degenerate at 37.3% hybrid censoring ($\hat\beta_1 \approx -20$). The Burr+EM-GMM trapezoidal grid remains stable across all three censoring levels but requires explicit $a \geq 1$ regularisation to prevent $S_\text{Burr}\to 1$ degeneracy. The ASSE ranking under heavy censoring reverses significantly relative to the uncensored case.

6. **Convergence of MCEM**: the MC log-likelihood oscillates even at M=200; convergence is assessed by parameter stability rather than the standard EM monotone likelihood criterion. A Rao-Blackwellised estimator or larger M would be needed for formal convergence guarantees.

### 10.5 Future Directions

- **EM-GMM regularisation**: the Mode A ($\sigma \geq 0.15$) constraint in `rfl_burr_em.py` is heuristic. A principled approach — penalised likelihood, minimum-description-length, or LOO validation — is needed to confirm ASSE=4.09 generalises out-of-sample.
- **Profile SE for SEV+INLA**: extend `rfl_profile.py` Profile Likelihood to the 5D SEV+INLA parametrisation for calibrated confidence intervals (complementing the Laplace-approximation SEs in Direction B).
- **Full HMC**: replace the Laplace approximation in Direction B with full NUTS/HMC (PyMC or NumPyro) for more accurate posterior sampling, especially in the non-Gaussian tails of $\sigma_\Delta$.
- **K-mixture with LOO selection**: use PSIS-LOO (Direction A) as the model selection criterion for K in Direction C, instead of AIC/BIC — this would give a truly out-of-sample justified K.
- **Censoring regularisation for EM-GMM**: a proper Gamma hyperprior on $a$ (e.g., $a \sim \text{Gamma}(\alpha_0, \beta_0)$ with $\alpha_0 > 1$) would provide principled regularisation instead of the heuristic $a \geq 1$ bound.
- **IS-corrected ILA for GH bias quantification**: validate the accuracy of the per-observation 9-pt Gauss–Hermite approximation in SEV+INLA using the importance-sampling correction proposed by Lai, Margossian & Sheldon (2026, arXiv:2605.20345). The single vs multiple Laplace gap is only 0.02 nats on the P&M data, suggesting the 1D $\Delta_i$ integral is already well-approximated — but IS correction could systematically quantify any residual bias, particularly in the near-degenerate $\hat\sigma_\Delta \approx 0.033$ regime where the LogNormal $g(\Delta)$ is extremely concentrated.

## 11. Structural Extensions (2026-05-27)

Three structural improvements beyond the baseline methods above, implemented in `rfl_loo.py`, `rfl_bayes_theta.py`, and `rfl_sev_inla_mix.py`.

### 11.1 Direction A — PSIS-LOO: Separating In-Sample Fit from Prediction

**File:** `rfl_loo.py`  
**Motivation:** All reported ASSE figures are in-sample; the rankings might be artefacts of overfitting rather than genuine predictive ability.

**Method:** Influence-function LOO — $\hat\theta_{-i} \approx \hat\theta + H_\text{info}^{-1} \nabla\ell_i(\hat\theta)$ — avoids $n=75$ refits. The "Prior-LOO" prediction for observation $i$ uses $\hat\theta_{-i}$ but replaces the per-obs posterior mean of $\Delta_i$ with the marginal prior mean:

$$\hat y_i^{\text{LOO}} = \mathbb{E}_{g(\Delta;\,\hat\theta_{-i})}\bigl[\mu(S_i,\Delta)\bigr] - \hat\sigma\gamma$$

This is the prediction a new specimen at stress $S_i$ would get — no conditioning on $y_i$.

**Key findings:**

| Method | In-sample ASSE | Prior-LOO ASSE | Sharpening Bonus |
|--------|:--------------:|:--------------:|:----------------:|
| SEV+INLA | 5.76 | ~40.4 | **+34.6** |
| Normal+NPMLE | 33.87 | ~41.5 | +7.6 |
| S-conditional baseline | — | ~40 | 0 |

The **Posterior Sharpening Bonus** = In-sample ASSE − Prior-LOO ASSE measures the benefit of conditioning on $y_i$ itself to estimate $\Delta_i$.  All methods achieve **Prior-LOO ASSE ≈ 40**, identical to the stress-level marginal mean — no method beats the mean for predicting a new specimen.

> **Implication:** The Fitted MAE = 5.76 for SEV+INLA is not a prediction-quality metric; it measures how well the model reconstructs observations it was fitted on (in-sample double-dipping). The $\hat\sigma_\Delta \approx 0.033$ near-degeneracy makes the posterior of $\Delta_i \mid y_i$ extremely sharp, concentrating the fitted value close to $y_i$. This is why Fitted MAE $\ll$ rank-ASSE: the former rewards posterior sharpness, the latter tests marginal calibration.

### 11.2 Direction B — Posterior Predictive ASSE ($\theta$ Integrated Out)

**File:** `rfl_bayes_theta.py`  
**Motivation:** Replace plugin estimate $\text{ASSE}(\hat\theta)$ with full Bayesian integration:

$$\text{ASSE}_\text{Bayes} = \mathbb{E}_{\theta \sim p(\theta|y)}\!\left[\sum_i\bigl|\hat y_i(\theta) - y_i\bigr|\right]$$

**Method:** Laplace approximation $p(\theta|y) \approx \mathcal{N}(\hat\theta, H_\text{obs}^{-1})$; numerical Hessian of the Laplace-GH loglik; $M=2000$ posterior samples.

**Posterior standard errors** (from $H_\text{obs}^{-1}$):

| Parameter | MLE | Post. SD | 95% CI |
|-----------|-----|----------|--------|
| $\beta_0$ | −9.370 | 0.299 | [−9.96, −8.78] |
| $\beta_1$ | −8.534 | 0.363 | [−9.24, −7.83] |
| $\sigma$ | 0.190 | 0.045 | [0.121, 0.299] |
| $\mu_\Delta$ | −0.644 | 0.023 | [−0.688, −0.599] |
| $\sigma_\Delta$ | 0.036 | 0.004 | [0.029, 0.045] |

**ASSE comparison:**

| Metric | Value |
|--------|------:|
| $\text{ASSE}_\text{plugin}$ (Laplace GH, $\hat\theta$ fixed) | 5.76 |
| $\text{ASSE}_\text{Bayes}$ mean | ~10.9 |
| $\text{ASSE}_\text{Bayes}$ 95% CI | [8.6, 15.0] |
| Inflation from $\theta$-uncertainty | +~16% |

> **Implication:** Integrating out $\theta$ adds only ≈16% to the ASSE point estimate. The dominant uncertainty is not global parameter estimation but the per-observation latent variable $\Delta_i$. LOO ASSE ≈ 40 $\gg$ 5.76 — the in-sample reduction is driven by the $\Delta_i$ posterior, not $\theta$ SE.

### 11.3 Direction C — K-Component LogNormal Mixture for $g(\Delta)$

**File:** `rfl_sev_inla_mix.py`  
**Motivation:** The unimodal LogNormal $g(\Delta)$ may be misspecified (NPMLE suggests K=2–3 support points for Normal, K=3 for SEV). A K-component LogNormal mixture bridges the discrete NPMLE structure with the continuous posterior flexibility of INLA.

**Model:** $$g(\Delta) = \sum_{k=1}^K \pi_k \text{LogNormal}(\mu_k, \sigma_k^2)$$, SEV conditional, 9-pt GH per component.

**Results** (consistent fast-GH comparison across K):

| Model | params | log-lik | AIC | BIC | ASSE |
|-------|-------:|--------:|----:|----:|-----:|
| K=1 (LogNormal, re-opt.) | 5 | −72.53 | 155.1 | **166.7** | 8.26 |
| K=2 mixture | 8 | −70.31 | 156.6 | 175.2 | 11.76 |
| **K=3 mixture** | **11** | **−62.45** | **146.9** | 172.4 | **5.14** |

**K=2 degenerates** ($\hat\pi_2 = 0.013$): the NPMLE atoms at $\{0.532, 0.569\}$ differ by only 0.037 — insufficient separation for two LogNormal components under the SEV likelihood.

**K=3 component estimates:**

| Comp | $\hat\pi_k$ | $E[\Delta_k]$ | $\hat\sigma_{d,k}$ |
|------|:-----------:|:-------------:|:------------------:|
| 1 | 0.627 | 0.494 | 0.032 |
| 2 | 0.245 | 0.467 | 0.028 |
| 3 | 0.128 | 0.465 | 0.005 |

K=3 wins on AIC (ΔAIC = −8.2 vs K=1) and ASSE, but **BIC favours K=1** (BIC penalises 6 extra parameters; at $n=75$ this is decisive). Components 2 and 3 are nearly co-located ($E[\Delta] \approx 0.467$ vs $0.465$) — K=3 likely overfits, effectively acting as a 2-component model with a degenerate third atom.

> **Implication:** $g(\Delta)$ shows mild departure from unimodality (AIC-optimal K=3), consistent with the NPMLE BIC selecting K=3 for the SEV model. However, $n=75$ is insufficient to reliably distinguish K=2 from K=3. The Laplace-GH K=1 ASSE=5.76 remains the best calibrated estimate under the unimodal prior.

## Appendix A: File Structure

```
rfl-inla/
├── rfl_inla.py              # Normal + INLA-style Multiple Laplace (per-obs residual=8.63)
├── rfl_sev_inla.py          # SEV + INLA: SEV conditional + 9-pt GH (per-obs residual=5.76) ⭐
├── rfl_profile.py           # Semi-parametric EM + Profile SE (Normal + NPMLE)
├── rfl_sev.py               # SEV + NPMLE: Weibull conditional, BFGS M-step, Profile SE
├── rfl_sev_ksel.py          # K=1..4 selection: Normal+NPMLE vs SEV+NPMLE
├── rfl_mcem.py              # SEV + MCEM: LogNormal g(Delta), rejection sampling E-step
├── rfl_em.py                # Basic EM + BIC model selection
├── rfl_residuals_run.py     # Compute absolute residuals from fitted model
├── ssla_se.py               # SSLA-based SE: Direction A (ASSLA-EM) + B (SSLA-INLA)
├── rfl_burr.py              # Burr XII MLE v1 (4-param, degenerate b->0)
├── rfl_burr2.py             # Burr XII MLE v2 (5-param, stress-dependent prior, per-obs residual=15.92)
├── rfl_burr_inla.py         # Burr XII + INLA (6-param, 3-level hierarchy, per-obs residual=5.74)
├── rfl_burr_em.py           # Burr XII + EM-GMM prior (K-comp. Gaussian mix., per-obs residual=4.09) ⭐
├── rfl_chiu.py              # Chiu (2005) EIV reproduction + direct ASSE optimisation (4 methods) 🏆
├── rfl_crps.py              # Marginal CRPS (energy score) optimisation + KM-based censoring
├── rfl_compare_all.py       # All 6 models: ASSE(n=75) + ASSE_te (holdout) + ASSE(n=75) (P&M Table 3)
├── rfl_compare.py           # Lightweight model comparison utilities
├── rfl_asse_zscore.py       # Z-score ASSE: rank→LogNormal quantile→predicted lnY (no fitting)
├── rfl_holdout.py           # Holdout evaluation on a train/test split
├── rfl_loo.py               # Direction A: influence-function LOO, Prior-LOO ASSE ≈ 40 all methods
├── rfl_bayes_theta.py       # Direction B: Laplace posterior p(θ|y), ASSE_Bayes=10.9±1.6
├── rfl_sev_inla_mix.py      # Direction C: K-mixture LogNormal g(Δ), K=3 AIC-best (ΔAIC=−8.2)
├── rfl_sim.py               # Simulation studies (coverage, bias)
├── rfl_sim2.py              # Simulation studies v2 (extended scenarios)
├── data/
│   └── pascual_meeker_1999.csv   # n=75 fatigue dataset (log-life + stress)
├── requirements.txt
└── README.md
```

## Appendix B: Quick Start

```bash
pip install numpy scipy joblib

# ALL-MODEL holdout + E criterion comparison (P&M Table 3 rank-matched error)
python rfl_compare_all.py
# Outputs ASSE(n=75) (rank-matched quantile error) + ASSE(n=75) + ASSE_te for all 6 models

# 1. Normal + NPMLE (fast, closed-form M-step)
python rfl_profile.py

# 2. SEV + NPMLE (Weibull conditional, BFGS M-step, Profile SE)
python rfl_sev.py

# 3. K=1..4 selection: Normal vs SEV NPMLE
python rfl_sev_ksel.py

# 4. SEV + MCEM (continuous LogNormal g, rejection sampling, ASSE ≈ Chiu thesis)
python rfl_mcem.py

# 5. Normal + INLA-style Multiple Laplace (per-obs residual=8.63)
python rfl_inla.py

# 6. SEV + INLA: best in-sample result, per-obs residual=5.76
python rfl_sev_inla.py

# 7. Residual analysis
python rfl_residuals_run.py

# 8. SSLA-based SE estimation (Direction A: ASSLA-EM + Direction B: SSLA-INLA)
python ssla_se.py

# 9. Burr XII closed-form MLE v1 (degenerate — b->0, ASSE=3.59 artifactual)
python rfl_burr.py

# 10. Burr XII closed-form MLE v2 (stress-dependent prior, 5-param, ASSE=15.92)
python rfl_burr2.py

# 11. Burr XII + INLA: 3-level hierarchy, nested in SEV+INLA (a->inf), per-obs residual=5.74
python rfl_burr_inla.py

# 12. Burr XII + EM-GMM: K-component Gaussian mix. prior, best per-obs residual=4.09 (Mode A, K=1)
python rfl_burr_em.py

# Structural Extensions (2026-05-27)
# 13. Direction A — influence-function LOO; Prior-LOO ASSE ≈ 40 for all methods
python rfl_loo.py

# 14. Direction B — Laplace posterior p(θ|y); ASSE_Bayes = 10.9 ± 1.6
python rfl_bayes_theta.py

# 15. Direction C — K-mixture LogNormal g(Δ); K=3 AIC-best (ΔAIC=−8.2 vs K=1)
python rfl_sev_inla_mix.py

# 16. Z-score ASSE: rank→LogNormal Δ quantile→predicted lnY (no fitting, hardcoded params)
python rfl_asse_zscore.py

# Chiu (2005) EIV Reproduction + Direct ASSE Optimisation (2026-05-29)
# 17. Method A: reproduce thesis (z-ASSE=10.78, rank-ASSE=12.41)
#     Method B: direct rank-ASSE opt (rank-ASSE=12.24, new best)
#     Method C-1: OLS z-ASSE opt (z-ASSE=10.31)
#     Method C-2: LAD z-ASSE opt (z-ASSE=9.94, BEATS THESIS ← new champion)
python rfl_chiu.py

# 18. Marginal CRPS (energy score) optimisation; KM-based censoring framework
python rfl_crps.py
```

## References

### Fatigue Models & S-N Curves

1. **Bastenaire, F. A.** (1972). "New method for the statistical evaluation of constant stress amplitude fatigue-test results." In *Probability Aspects of Fatigue* (ASTM STP 511), ed. R. A. Heller. Philadelphia: ASTM, pp. 3–28.

2. **Castillo, E., Fernández-Canteli, A., Esslinger, V., and Thürlimann, B.** (1985). "Statistical model for fatigue analysis of wires, strands and cables." *IABSE Proceedings* P-82/85. Zurich: IABSE, pp. 1–40.

3. **Castillo, E. and Hadi, A. S.** (1995). "Modeling lifetime data with application to fatigue models." *Journal of the American Statistical Association*, **90**, 1041–1054.

4. **Little, R. E. and Ekvall, J. C.** (eds.) (1981). *Statistical Analysis of Fatigue Data* (ASTM STP 744). Philadelphia: ASTM.

5. **Pascual, F. G. and Meeker, W. Q.** (1999). "Estimating fatigue curves with the random fatigue-limit model." *Technometrics*, **41**(4), 277–302.

6. **Pascual, F. G.** (2003). "The random fatigue-limit model in multi-factor experiment." *Journal of Statistical Computation and Simulation*, **10**, 733–752.

7. **Spindel, J. E. and Haibach, E.** (1981). "Some considerations in the statistical determination of the shape of the S-N curves." In *Statistical Analysis of Fatigue Data* (ASTM STP 744), eds. Little & Ekvall. Philadelphia: ASTM, pp. 89–113.

### Reliability Data & Extreme Value Distributions

8. **Meeker, W. Q. and Escobar, L. A.** (1998). *Statistical Methods for Reliability Data*. New York: John Wiley & Sons.
    > Standard reference for Weibull/SEV lifetime models, accelerated life testing, and the physical basis of the smallest extreme value distribution in reliability engineering.

9. **Fisher, R. A. and Tippett, L. H. C.** (1928). "Limiting forms of the frequency distribution of the largest or smallest member of a sample." *Proceedings of the Cambridge Philosophical Society*, **24**(2), 180–190.
    > Theoretical foundation of extreme value theory: the limiting distribution of the minimum of many i.i.d. random variables is the Gumbel-min (SEV) distribution — the weakest-link justification for the SEV conditional in the RFL model.

### Mixture Models & Semi-parametric Theory

10. **Lindsay, B. G.** (1983). "The geometry of mixture likelihoods: A general theory." *Annals of Statistics*, **11**(1), 86–94.

11. **McLachlan, G. J. and Peel, D.** (2000). *Finite Mixture Models*. Wiley Series in Probability and Statistics. New York: John Wiley & Sons.
    > Standard reference for finite mixture model theory, EM algorithm for mixtures, and identifiability. Basis for the K-component LogNormal mixture $g(\Delta)$ in Direction C (`rfl_sev_inla_mix.py`).

12. **Murphy, S. A. and van der Vaart, A. W.** (2000). "On profile likelihood." *Journal of the American Statistical Association*, **95**(450), 449–465.

13. **Teicher, H.** (1963). "Identifiability of finite mixtures." *Annals of Mathematical Statistics*, **34**(4), 1265–1270.

### INLA — Integrated Nested Laplace Approximation

14. **Rue, H., Martino, S., and Chopin, N.** (2009). "Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations." *Journal of the Royal Statistical Society, Series B*, **71**(2), 319–392.
    > Two-level Laplace framework adopted here: inner per-latent-variable approximation + outer hyperparameter search.

### Monte Carlo EM

15. **Wei, G. C. G. and Tanner, M. A.** (1990). "A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms." *Journal of the American Statistical Association*, **85**(411), 699–704.
    > Original MCEM paper: replaces the intractable E-step expectation with a Monte Carlo average over samples from the complete-data posterior. Forms the theoretical basis of `rfl_mcem.py`.

### Heuristic Learning — Simulated Annealing & Dual Annealing

16. **Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P.** (1983). "Optimization by simulated annealing." *Science*, **220**(4598), 671–680. DOI: 10.1126/science.220.4598.671
    > Original SA: temperature-adaptive random walk for global optimisation of non-convex objectives.

17. **Tsallis, C. and Stariolo, D. A.** (1996). "Generalized simulated annealing." *Physica A*, **233**, 395–406.
    > Theoretical basis of dual annealing: Tsallis statistics replace the Boltzmann acceptance criterion.

18. **Xiang, Y., Sun, D. Y., Fan, W., and Gong, X. G.** (1997). "Generalized simulated annealing algorithm and its application to the Thomson model." *Physics Letters A*, **233**, 216–220.
    > Practical GSA algorithm; forms the core of `scipy.optimize.dual_annealing`.

19. **Xiang, Y. and Gong, X. G.** (2000). "Efficiency of generalized simulated annealing." *Physical Review E*, **62**, 4473.
    > Efficiency analysis of GSA; used by SciPy's dual-annealing implementation.

### Local Optimisation

20. **Nelder, J. A. and Mead, R.** (1965). "A simplex method for function minimization." *The Computer Journal*, **7**(4), 308–313.
    > Derivative-free downhill simplex (Stage 3 polish in optimisation pipeline).

### Error-in-Variables & Measurement Error

21. **Fuller, W. A.** (1987). *Measurement Error Models*. New York: John Wiley & Sons.

### Self-Supervised Uncertainty Quantification

22. **Rodemann, J., Marquard, A., Augustin, T., and Caprio, M.** (2026). "Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification." *Transactions on Machine Learning Research*. arXiv:2605.12208.
    > SSLA/ASSLA: bypass parameter posterior, directly approximate posterior predictive by refitting on self-predicted data. Deterministic, sampling-free. Applied in `ssla_se.py` to fix the Louis 7.7× underestimation and provide full 5D UQ.

### Thesis

23. **Chiu, C.-H.** (2005). *A Family of Bivariate Distributions With Some Applications to Statistical Inferences*. M.Sc. thesis, Graduate Institute of Management Sciences, Tamkang University (Chapter 2: Random Fatigue-Limit Model via error-in-measurement regression, ASSE = 10.80).

### Leave-One-Out Cross-Validation & Influence Analysis

24. **Vehtari, A., Gelman, A., and Gabry, J.** (2017). "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing*, **27**(5), 1413–1432. DOI: 10.1007/s11222-016-9696-4
    > PSIS-LOO: uses Pareto-smoothed importance sampling to approximate LOO-CV from a single MCMC fit. Motivation for Direction A (`rfl_loo.py`): replaces $n=75$ refits with influence-function approximation $\hat\theta_{-i} \approx \hat\theta + H^{-1}\nabla\ell_i(\hat\theta)$.

25. **Cook, R. D. and Weisberg, S.** (1982). *Residuals and Influence in Regression*. New York: Chapman and Hall.
    > Classic treatment of influence functions and deletion diagnostics. The one-step influence approximation $\hat\theta_{-i} \approx \hat\theta + H_\text{info}^{-1}\nabla\ell_i(\hat\theta)$ used in Direction A derives from the standard leave-one-out influence formula (Cook's distance framework).

### Bayesian Posterior Approximation

26. **Tierney, L. and Kadane, J. B.** (1986). "Accurate approximations for posterior moments and marginal densities." *Journal of the American Statistical Association*, **81**(393), 82–86.
    > Laplace approximation to the posterior: $p(\theta|y) \approx \mathcal{N}(\hat\theta, H_\text{obs}^{-1})$ where $H_\text{obs} = -\partial^2\log p(\theta|y)/\partial\theta^2\big|_{\hat\theta}$. Theoretical basis for Direction B (`rfl_bayes_theta.py`), which numerically computes the $5\times5$ observed Hessian of the Laplace-GH log-likelihood to obtain posterior standard errors and the $M=2000$ sample Monte Carlo estimate of $\text{ASSE}_\text{Bayes}$.

### Kaplan–Meier Estimator & Censored Data Plotting Positions

27. **Kaplan, E. L. and Meier, P.** (1958). "Nonparametric estimation from incomplete observations." *Journal of the American Statistical Association*, **53**(282), 457–481. DOI: 10.1080/01621459.1958.10501452
    > Foundation of the Kaplan–Meier product-limit estimator. Used in `rfl_chiu.py` (`build_km_z()`) to compute plotting positions $p_t^{\text{KM}} = (1 - \hat{S}_{\text{KM}}(t)) - 0.5/n$ (Hazen correction) for right-censored observations, replacing within-group sample z-scores with $z_t = \Phi^{-1}(p_t^{\text{KM}})$. Supports Type I, Type II, hybrid, and progressive censoring under a unified interface.

### Proper Scoring Rules & Continuous Ranked Probability Score

28. **Gneiting, T. and Raftery, A. E.** (2007). "Strictly proper scoring rules, prediction, and estimation." *Journal of the American Statistical Association*, **102**(477), 359–378. DOI: 10.1198/016214506000001437
    > Theoretical framework for proper scoring rules. The **CRPS** (Continuous Ranked Probability Score) is the integral of the Brier score over all thresholds: $\text{CRPS}(F, y) = \int [F(t) - \mathbf{1}(t \geq y)]^2 dt$. For a mixture predictive distribution, the marginal CRPS is computed via the energy-score decomposition: $\text{CRPS}(F_W, y) = \mathbb{E}_\Delta[|X - y|] - \tfrac{1}{2}\mathbb{E}_{\Delta,\Delta'}[|X - X'|]$, both terms evaluated by 40-pt Gauss–Hermite over the LogNormal prior in `rfl_crps.py`. The naive average of conditional CRPSes yields a Jensen gap — the correct marginal CRPS requires the full energy-score form.

### Least Absolute Deviations & Quantile Regression

29. **Koenker, R. and Bassett, G.** (1978). "Regression quantiles." *Econometrica*, **46**(1), 33–50.
    > Foundational paper for LAD (least absolute deviations) / $L_1$ regression as the special case of quantile regression at $\tau = 0.5$. Method C-2 in `rfl_chiu.py` applies LAD to fit $(\beta_0, \beta_1)$ given the within-group z-score mapped fatigue limits $\hat\Delta_t = \exp(\mu_\Delta + \sigma_\Delta z_t)$. Because the z-ASSE metric is a sum of absolute errors (SAE), LAD is the natural minimiser — OLS (minimising SSE) used in the original Chiu (2005) thesis is suboptimal for this metric. Combined with Nelder-Mead over $(\mu_\Delta, \sigma_\Delta)$, C-2 achieves z-ASSE = 9.94, an 8% improvement over the thesis value of 10.80.

## Licence

MIT
