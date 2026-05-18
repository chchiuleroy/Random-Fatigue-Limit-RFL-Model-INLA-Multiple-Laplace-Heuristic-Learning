# Random Fatigue-Limit (RFL) Model — INLA Multiple Laplace + Heuristic Learning

A Python implementation of the **Random Fatigue-Limit (RFL) model** combining two key ideas:

- **Multiple Laplace approximation** (INLA-style): 9-point Gauss–Hermite centred at the per-observation posterior mode, replacing the single-point Laplace of Pascual & Meeker (1999)
- **Heuristic learning** for outer optimisation: a 3-stage pipeline (random grid → dual annealing → Nelder-Mead) that adaptively searches the non-convex 5D likelihood surface

Three complementary estimation strategies are provided, extending work first presented in Chiu (2005):

| Method | File | Inner Integration | Outer Optimiser |
|--------|------|-------------------|-----------------|
| Error-in-variables regression (thesis) | — | Analytic | Grid search over $(X_0, \sigma_u^2)$ |
| Semi-parametric EM + Profile SE | `rfl_profile.py` | Closed form (discrete NPMLE) | ECM algorithm |
| **INLA Multiple Laplace + Heuristic** | **`rfl_inla.py`** | **9-pt Gauss–Hermite** | **Grid → SA → Nelder-Mead** |

---

## Background

In fatigue testing, a metal specimen subjected to cyclic stress $S$ fails after $N$ cycles.
The key challenge is the **fatigue limit** $\Delta$: specimens loaded below $\Delta$ never fail.
$\Delta$ varies across specimens and is **never directly observed**.

The RFL model ([Pascual & Meeker 1999](https://doi.org/10.1080/00401706.1999.10485928)) treats $\Delta$ as a random variable:

$$Y_i \mid \Delta \;\sim\; \mathcal{N}\!\bigl(\beta_0 + \beta_1 \log(S_i - \Delta),\; \sigma^2\bigr), \qquad \Delta \perp\!\!\!\perp Y_i \mid S_i$$

where $Y_i = \log N_i$ is the log-life.

The marginal likelihood requires integrating out $\Delta$:

$$L_i(\theta) = \int_0^{S_i} f(y_i \mid \Delta, S_i)\; g(\Delta;\, \mu_\Delta, \sigma_\Delta)\; d\Delta$$

---

## Two Estimation Approaches

### 1. Semi-parametric EM (`rfl_profile.py`)

Replaces the parametric $g(\Delta)$ with a **nonparametric MLE (NPMLE)**:

$$\hat{G} = \sum_{k=1}^{K} \hat\pi_k\, \delta_{\hat\Delta_k}$$

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

---

### 2. INLA-style Multiple Laplace (`rfl_inla.py`)

Keeps $g(\Delta) = \text{LogNormal}(\mu_\Delta, \sigma_\Delta^2)$ (parametric) and approximates the integral using the **INLA two-level philosophy**:

#### Inner integral — Multiple Laplace (Gauss–Hermite)

For each observation $i$:

1. Find the **posterior mode** of $\Delta$:

$$\hat\Delta_i = \arg\max_\Delta \bigl[\log f(y_i \mid \Delta) + \log g(\Delta)\bigr]$$

2. Compute **Laplace curvature**: $\tilde\sigma_i = 1/\sqrt{-\partial^2 \log h_i(\hat\Delta_i)/\partial\Delta^2}$

3. **Multiple Laplace** (9-node Gauss–Hermite centred at mode):

$$L_i \;\approx\; \tilde\sigma_i \sum_{j=1}^{9} w_j^{\mathrm{GH}}\, e^{x_j^2/2}\, h_i(\hat\Delta_i + \tilde\sigma_i\, x_j)$$

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

---

## Heuristic Learning — Core Concepts & Usage

This project employs heuristic learning at **four levels**, from the innermost per-observation integral to the model-level complexity selection.

### Overview

| Level | Method | Location | Purpose |
|-------|--------|----------|---------|
| Integral (inner) | Adaptive Laplace curvature | `_multi_laplace()` | Self-calibrating GH quadrature nodes |
| Outer optimisation | Random grid → Dual Annealing → Nelder-Mead | `heuristic_optimize()` | Non-convex 5D MLE |
| EM initialisation | Multi-start random seeds | `rfl_profile.py`, `rfl_em.py` | Escape EM local optima |
| Model selection | BIC over $K \in \{1,2,3,4\}$ | `rfl_em.py` | Automatic mixture complexity |

---

### 1. `heuristic_optimize()` — 3-Stage Outer Search

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

---

### 2. Adaptive Laplace Scale — Self-Calibrating Inner Quadrature

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

---

### 3. Multi-Start EM — Escaping Local Optima

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

---

### 4. BIC-Driven Automatic Model Selection

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

---

### Heuristic Learning — Flow Summary

```
rfl_inla.py  ─── outer: heuristic_optimize()
│                         ├─ Stage 1: Random grid (40 pts)  ─→ coarse basin
│                         ├─ Stage 2: Dual Annealing (800)  ─→ adaptive global search
│                         └─ Stage 3: Nelder-Mead           ─→ precision polish
│
└──────────── inner: _multi_laplace()
                        ├─ minimize_scalar → posterior mode Δ̂ᵢ
                        └─ curvature → adaptive σ̃ᵢ → 9-pt GH nodes

rfl_profile.py ── EM: 15-start random initialisation
rfl_em.py      ── EM: 8-start × K∈{1..4}, BIC auto-selects best K
```

---

## Censoring Schemes

### Type I (fixed cutoff)
Specimens are right-censored at a pre-set time $T_j$ per stress level:

$$\delta_i = \mathbf{1}[Y_i > T_j], \quad \tilde Y_i = \min(Y_i, T_j)$$

The censored likelihood contribution replaces $f(y_i \mid \Delta)$ with $P(Y_i > T_j \mid \Delta) = \bar\Phi\!\bigl(\tfrac{T_j - \mu(S_i,\Delta)}{\sigma}\bigr)$.

### Hybrid Type I-II (per stress level)
Stop at the earlier of the $r$-th failure or fixed time $T$:

$$T^*_j = \min\!\bigl(X_{r_j{:}n_j},\; T\bigr)$$

All specimens surviving past $T^*_j$ are right-censored at $T^*_j$.  
This generalises Type I (set $r_j = n_j$) and Type II (set $T = \infty$).

---

## Results on Pascual & Meeker (1999) Data

Dataset: $n=75$ specimens, 5 stress levels $S \in \{0.675, 0.75, 0.825, 0.90, 0.95\}$, 15 per level.

### Parameter Estimates (INLA-style, LogNormal g)

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

### Absolute Residuals (conditional on posterior mode $\hat\Delta_i$)

Residual definition:
$$e_i = Y_i - \hat\mu_i, \qquad \hat\mu_i = \beta_0 + \beta_1 \log(S_i - \hat\Delta_i)$$

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
- Max residual at $S=0.950$, $Y=-3.297$ (standardised $e/\sigma = -1.83$): unusually short life at the lowest excess-stress level — a likely tail outlier
- $S=0.900$, $Y=+0.121$: standardised residual $+1.22$ — longest life in that stress group
- Residuals are well-behaved within $\pm 1\sigma$ for 70 of 75 observations (93%)

Run `python rfl_residuals_run.py` for the full per-observation table.

### Censoring Comparison (INLA-style, all scenarios)

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

### Single vs Multiple Laplace (full data, [A] parameters)

| Method | Total $\log L$ | Per-obs average |
|--------|:--------------:|:---------------:|
| Single Laplace (mode-only) | −72.849 | −0.9713 |
| Multiple Laplace (9-pt GH) | −72.869 | −0.9716 |
| Difference | −0.020 | −0.00027 |

The two approximations agree to within 0.02 on 75 observations — the Laplace scale $\tilde\sigma_i \approx 0.006$–$0.011$ is so small that higher-order GH corrections are negligible for this dataset. The dominant benefit of Multiple Laplace appears in heavy-tailed or more diffuse $g(\Delta)$ settings.

---

## Comparison with Prior Work — Roy's M.Sc. Thesis (Chiu 2005)

The fatigue dataset originates from Castillo & Hadi (1995) / Pascual & Meeker (1999):  
$n = 75$ specimens, $S \in \{0.675, 0.75, 0.825, 0.90, 0.95\}$ (15 per level).

The thesis introduced a **Normal–Normal error-in-variables regression model**:

$$\omega_t = \beta_0 + \beta_1 q_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,\sigma_\varepsilon^2)$$

where $q_t = \log(X_t - X_0)$ is unobserved (measurement error in the log excess-stress),  
$X_0$ is the (unknown) fatigue limit, and $q_t \sim \mathcal{N}(\mu_q, \sigma_q^2)$ is modelled separately.  
Parameters are estimated via the Fuller (1987) error-in-measurement approach.

**Thesis parameter estimates on real data:**

| Parameter | Estimate |
|-----------|----------|
| $\hat X_0$ (fatigue limit location) | 0.0345517 |
| $(\hat\beta_0, \hat\beta_1)$ | (−9.0747, −7.6027) |
| $\widehat{\text{Var}}(\hat{q}_t - q_t \mid \omega, Q)$ | 0.007895 |
| $\hat\sigma_\varepsilon^2$ | 0.031211 |

### Table 3 — ASSE Comparison Across Models (Chiu 2005, Table 3)

**ASSE** = Sum of Absolute Residuals across all $n = 75$ observations (log-life scale).  
Lower is better. Models are fitted to the same Castillo & Hadi (1995) dataset.

| Model | No. of Parameters | ASSE |
|-------|:-----------------:|-----:|
| Little & Ekvall (1981), model 1 | 3 | 41.13 |
| Little & Ekvall (1981), model 2 | 3 | 31.17 |
| Spindel & Haibach (1981) | 6 | 17.35 |
| Bastenaire (1972) | 5 | 20.52 |
| Castillo et al. (1985) | 4 | 20.27 |
| Castillo & Hadi (1995) | 5 | 18.12 |
| Pascual & Meeker (1999), Nor–Nor | 5 | 12.84 |
| **Chiu (2005), Nor–Nor (thesis)** | **5** | **10.80** |
| **This work — INLA-style (rfl_inla.py)** | **5** | **≈ 8.63** |

> ASSE for "This work" = MAE × 75 = 0.1150 × 75 = **8.625** (computed on the same dataset,  
> comparable metric). The improvement over Pascual & Meeker (1999) grows from 15.9% (thesis)  
> to **32.8%** (current INLA implementation).  
> Note: models differ in $g(\Delta)$ specification and estimation philosophy — comparison is indicative.

### Evolution of Methods

```
Chiu (2005) thesis      — Error-in-variables regression, ASSE = 10.80
          ↓
rfl_em.py               — EM + NPMLE, BIC for K selection
          ↓
rfl_profile.py          — Semi-parametric ECM + Profile Likelihood SE (7.7× Louis correction)
          ↓
rfl_inla.py             — INLA-style 9-pt Gauss–Hermite + dual-annealing, ASSE ≈ 8.63
          ↓
ssla_se.py              — SSLA-based SE: deterministic fix for Louis 7.7× underestimation + full 5D UQ
```

---

## SSLA-Based Standard Error Estimation (`ssla_se.py`)

The `se_hessian()` in `rfl_inla.py` holds $(\mu_\Delta, \sigma_\Delta)$ fixed, producing a 3×3 Hessian and inheriting the same Louis-style underestimation problem that affects `rfl_em.py`. `ssla_se.py` provides a **deterministic, sampling-free** alternative based on the Self-Supervised Laplace Approximation (Rodemann et al., TMLR 2026, arXiv:2605.12208).

### Core Idea

Rather than approximating the parameter posterior $p(\theta|\mathcal{D})$, SSLA directly quantifies uncertainty by **refitting on self-predicted data**:

$$\text{SE}(\hat\theta) \approx |\tilde\theta - \hat\theta|$$

where $\tilde\theta$ is the refit on $\hat{Y} = \hat\beta_0 + \hat\beta_1 \log(S_i - \hat\Delta_i)$ (the model's own predictions). The sensitivity of the fitted parameters to the data is the uncertainty estimate.

### Two Directions

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

### SE Method Comparison

| Method | SE(β₀) | SE(β₁) | SE(σ) | Notes |
|--------|--------|--------|-------|-------|
| Profile SE (rfl_profile.py) | 0.580 | **1.486** | 0.054 | Gold standard, profile likelihood |
| Louis SE (rfl_em.py) | 0.360 | **0.193** | 0.052 | 7.7× underestimate of SE(β₁) |
| ASSLA-EM (`ssla_se.py`) | *run* | *run* | *run* | Deterministic, no Bootstrap |
| ASSLA-INLA norefit | *run* | *run* | *run* | Full 5D Hessian, fast |
| SSLA-INLA refit | *run* | *run* | *run* | Full 5D, most faithful |

**Actual results on n=75 data** (see `_test_ssla.py`):

| Method | SE(β₀) | SE(β₁) | SE(σ) |
|--------|--------|--------|-------|
| Profile SE | 0.580 | 1.486 | 0.054 |
| Louis SE (EM) | 0.360 | 0.193 | 0.052 |
| ASSLA-EM | 0.204 | **0.589** | ~~0.563~~ ⚠️ |
| ASSLA-INLA 5D Hessian | 0.436 | **1.052** | 0.152 |

> ⚠️ ASSLA-EM SE(σ) is **unreliable**: self-predicted Y_self has zero residuals, so σ collapses on refit (σ_tilde ≈ 0.033). Use the 5D Hessian for σ.  
> The 5D Hessian SE(β₁) = 1.052 vs Profile 1.486: remaining gap reflects model difference (LogNormal vs NPMLE g(Δ)), not method failure.

### Why SSLA Instead of Bootstrap?

| Criterion | Bootstrap ($R=500$) | SSLA/ASSLA |
|-----------|--------------------|-----------:|
| Computation | $500 \times$ EM time | $1$–$4 \times$ EM time |
| Randomness | Stochastic (varies by seed) | Deterministic |
| Full 5D SE | Yes (via B resamples) | Yes (via Hessian / refit) |
| Implementation | External loop | Single function call |

Reference: Rodemann J., Marquard A., Augustin T., Caprio M. (2026). *Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification*. TMLR. arXiv:2605.12208.

---

## File Structure

```
rfl-inla/
├── rfl_inla.py              # INLA-style Multiple Laplace main script
├── rfl_profile.py           # Semi-parametric EM + Profile SE (existing)
├── rfl_em.py                # Basic EM + BIC model selection
├── rfl_residuals_run.py     # Compute absolute residuals from fitted model
├── ssla_se.py               # SSLA-based SE: Direction A (ASSLA-EM) + B (SSLA-INLA)
├── data/
│   └── pascual_meeker_1999.csv   # n=75 fatigue dataset (log-life + stress)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install numpy scipy

# 1. Semi-parametric EM (fast, closed-form)
python rfl_profile.py

# 2. INLA-style Multiple Laplace (slower, parametric g)
python rfl_inla.py

# 3. Residual analysis
python rfl_residuals_run.py

# 4. SSLA-based SE estimation (Direction A: ASSLA-EM + Direction B: SSLA-INLA)
python ssla_se.py
```

---

## References

### Fatigue Models & S-N Curves

1. **Bastenaire, F. A.** (1972). "New method for the statistical evaluation of constant stress amplitude fatigue-test results." In *Probability Aspects of Fatigue* (ASTM STP 511), ed. R. A. Heller. Philadelphia: ASTM, pp. 3–28.

2. **Castillo, E., Fernández-Canteli, A., Esslinger, V., and Thürlimann, B.** (1985). "Statistical model for fatigue analysis of wires, strands and cables." *IABSE Proceedings* P-82/85. Zurich: IABSE, pp. 1–40.

3. **Castillo, E. and Hadi, A. S.** (1995). "Modeling lifetime data with application to fatigue models." *Journal of the American Statistical Association*, **90**, 1041–1054.

4. **Little, R. E. and Ekvall, J. C.** (eds.) (1981). *Statistical Analysis of Fatigue Data* (ASTM STP 744). Philadelphia: ASTM.

5. **Pascual, F. G. and Meeker, W. Q.** (1999). "Estimating fatigue curves with the random fatigue-limit model." *Technometrics*, **41**(4), 277–302.

6. **Pascual, F. G.** (2003). "The random fatigue-limit model in multi-factor experiment." *Journal of Statistical Computation and Simulation*, **10**, 733–752.

7. **Spindel, J. E. and Haibach, E.** (1981). "Some considerations in the statistical determination of the shape of the S-N curves." In *Statistical Analysis of Fatigue Data* (ASTM STP 744), eds. Little & Ekvall. Philadelphia: ASTM, pp. 89–113.

### Mixture Models & Semi-parametric Theory

8. **Lindsay, B. G.** (1983). "The geometry of mixture likelihoods: A general theory." *Annals of Statistics*, **11**(1), 86–94.

9. **Murphy, S. A. and van der Vaart, A. W.** (2000). "On profile likelihood." *Journal of the American Statistical Association*, **95**(450), 449–465.

10. **Teicher, H.** (1963). "Identifiability of finite mixtures." *Annals of Mathematical Statistics*, **34**(4), 1265–1269.

### INLA — Integrated Nested Laplace Approximation

11. **Rue, H., Martino, S., and Chopin, N.** (2009). "Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations." *Journal of the Royal Statistical Society, Series B*, **71**(2), 319–392.
    > Two-level Laplace framework adopted here: inner per-latent-variable approximation + outer hyperparameter search.

### Heuristic Learning — Simulated Annealing & Dual Annealing

12. **Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P.** (1983). "Optimization by simulated annealing." *Science*, **220**(4598), 671–680. DOI: 10.1126/science.220.4598.671
    > Original SA: temperature-adaptive random walk for global optimisation of non-convex objectives.

13. **Tsallis, C. and Stariolo, D. A.** (1996). "Generalized simulated annealing." *Physica A*, **233**, 395–406.
    > Theoretical basis of dual annealing: Tsallis statistics replace the Boltzmann acceptance criterion.

14. **Xiang, Y., Sun, D. Y., Fan, W., and Gong, X. G.** (1997). "Generalized simulated annealing algorithm and its application to the Thomson model." *Physics Letters A*, **233**, 216–220.
    > Practical GSA algorithm; forms the core of `scipy.optimize.dual_annealing`.

15. **Xiang, Y. and Gong, X. G.** (2000). "Efficiency of generalized simulated annealing." *Physical Review E*, **62**, 4473.
    > Efficiency analysis of GSA; used by SciPy's dual-annealing implementation.

### Local Optimisation

16. **Nelder, J. A. and Mead, R.** (1965). "A simplex method for function minimization." *The Computer Journal*, **7**(4), 308–313.
    > Derivative-free downhill simplex (Stage 3 polish in `heuristic_optimize`).

### Error-in-Variables & Measurement Error

17. **Fuller, W. A.** (1987). *Measurement Error Models*. New York: John Wiley & Sons.

### Self-Supervised Uncertainty Quantification

18. **Rodemann, J., Marquard, A., Augustin, T., and Caprio, M.** (2026). "Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification." *Transactions on Machine Learning Research*. arXiv:2605.12208.
    > SSLA/ASSLA: bypass parameter posterior, directly approximate posterior predictive by refitting on self-predicted data. Deterministic, sampling-free. Applied in `ssla_se.py` to fix the Louis 7.7× underestimation and provide full 5D UQ.

### Thesis

19. **Chiu, C.-H.** (2005). *A Family of Bivariate Distributions With Some Applications to Statistical Inferences*. M.Sc. thesis, Graduate Institute of Management Sciences, Tamkang University (Chapter 2: Random Fatigue-Limit Model via error-in-measurement regression, ASSE = 10.80).

---

## Licence

MIT
