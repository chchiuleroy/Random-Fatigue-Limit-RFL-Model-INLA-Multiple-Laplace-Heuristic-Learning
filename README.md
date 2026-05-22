# Random Fatigue-Limit (RFL) Model — INLA Multiple Laplace + Heuristic Learning

A Python implementation of the **Random Fatigue-Limit (RFL) model** combining two key ideas:

- **Multiple Laplace approximation** (INLA-style): 9-point Gauss–Hermite centred at the per-observation posterior mode, replacing the single-point Laplace of Pascual & Meeker (1999)
- **Heuristic learning** for outer optimisation: a 3-stage pipeline (random grid → dual annealing → Nelder-Mead) that adaptively searches the non-convex 5D likelihood surface

Six complementary estimation strategies are provided, extending work first presented in Chiu (2005):

| Method | File | Conditional $f(y\|\Delta)$ | Inner Integration | Outer Optimiser | ASSE |
|--------|------|--------------------------|-------------------|-----------------|-----:|
| Error-in-variables regression (thesis) | — | Normal | Analytic | Grid search | 10.80 |
| Semi-parametric EM + Profile SE | `rfl_profile.py` | Normal | Closed form (discrete NPMLE) | ECM | 33.87 |
| **SEV + NPMLE** | **`rfl_sev.py`** | **SEV** | **Closed form (discrete NPMLE)** | **ECM + BFGS** | 28.99 |
| **SEV + MCEM** | **`rfl_mcem.py`** | **SEV** | **Monte Carlo (M=200 rejection sampling)** | **BFGS + LogNormal MLE** | 10.89 |
| **INLA Multiple Laplace** | **`rfl_inla.py`** | **Normal** | **9-pt Gauss–Hermite** | **Grid → SA → NM** | 8.63 |
| **SEV + INLA** ⭐ | **`rfl_sev_inla.py`** | **SEV** | **9-pt Gauss–Hermite** | **Grid → SA → NM** | **5.76** |
| Burr XII MLE v1 (4-param, degenerate) | `rfl_burr.py` | SEV | Closed form (Gamma conjugate) | Grid → BFGS → NM | 3.59† |
| **Burr XII MLE v2 (5-param)** | **`rfl_burr2.py`** | **SEV** | **Closed form (stress-dep. prior)** | **Grid → BFGS → NM** | **15.92** |
| **Burr XII + INLA (6-param)** | **`rfl_burr_inla.py`** | **SEV** | **Burr XII inner + 9-pt GH outer** | **BFGS → NM** | **5.74** |
| **Burr XII + EM-GMM (Mode A, K=1)** ⭐ | **`rfl_burr_em.py`** | **SEV** | **Burr XII inner + trapezoidal grid** | **EM + L-BFGS-B** | **4.09** |

> † ASSE = 3.59 for `rfl_burr.py` is **artifactual**: the MLE drives $b \to 0$, making the fitted value trivially close to $y_i - \text{const}$. The 4-parameter model lacks stress-level differentiation ($\beta_1$ and $S_i$ are unidentifiable) and the "result" reflects a degenerate improper prior limit, not a genuine predictive gain.

---

## Background

In fatigue testing, a metal specimen subjected to cyclic stress $S$ fails after $N$ cycles.
The key challenge is the **fatigue limit** $\Delta$: specimens loaded below $\Delta$ never fail.
$\Delta$ varies across specimens and is **never directly observed**.

The RFL model ([Pascual & Meeker 1999](https://doi.org/10.1080/00401706.1999.10485928)) treats $\Delta$ as a random variable:

$$Y_i \mid \Delta \\sim\ \mathcal{N}\!\bigl(\beta_0 + \beta_1 \log(S_i - \Delta),\ \sigma^2\bigr), \qquad \Delta \perp\\perp Y_i \mid S_i$$

where $Y_i = \log N_i$ is the log-life.

The marginal likelihood requires integrating out $\Delta$:

$$L_i(\theta) = \int_0^{S_i} f(y_i \mid \Delta, S_i)\ g(\Delta\, \mu_\Delta, \sigma_\Delta)\ d\Delta$$

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

### 2. SEV + NPMLE (`rfl_sev.py`)

Replaces the Normal conditional with **Smallest Extreme Value (SEV)** — the log-lifetime distribution implied by the Weibull weakest-link model — while keeping the NPMLE discrete mixing distribution.

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

$$\hat\Delta_i = \Delta_{k^*_i}, \qquad e_i = Y_i - \bigl(\hat\beta_0 + \hat\beta_1 \log(S_i - \hat\Delta_i)\bigr)$$

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

---

### 3. INLA-style Multiple Laplace (`rfl_inla.py`)

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

### Cross-Method Censoring Comparison

All three continuous-prior methods evaluated under the same [A]/[B]/[C] scenarios.  
ASSE is computed on **uncensored observations only**; $n_\text{obs}$ varies by scenario.

| Method | Scenario | $n_\text{obs}$ | ASSE | $\hat\beta_1$ | $\hat\sigma$ | $\hat{a}$ / $\hat\sigma_d$ | Status |
|--------|----------|:--------------:|-----:|------:|------:|------:|--------|
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

The GH approach requires finding $\hat\Delta_i = \arg\max_\Delta [\log f(y_i|\Delta) + \log g(\Delta)]$ per observation. When $S=0.675$ specimens are mostly censored, this mode search becomes ill-conditioned — the censored log-likelihood $\log S(T_j|\Delta)$ is flat over $\Delta$ near the boundary, and the outer L-BFGS-B optimizer drifts $\beta_1$ toward $-\infty$ to compensate.

**Why EM-GMM remains stable:**

1. The trapezoidal grid integrates $\log f_\text{Burr}$ or $\log S_\text{Burr}$ over 400 grid points — no mode required
2. GMM posterior moments anchor the prior to observed $\Delta$ values
3. The $a \geq 1$ constraint prevents $S_\text{Burr}(y|\Delta) \to 1$ degeneracy (which would perfectly explain censored observations by making the specimen "never fail")

> Note: [B] ASSE for Burr+EM-GMM (4.92) is slightly worse than SEV+INLA (5.29) when expressed as per-observation MAE (4.92/60=0.082 vs 5.29/60=0.088 — actually Burr+EM-GMM is still better). Both evaluate on the same 60 uncensored observations.

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

| Model | No. of Parameters | ASSE | Residual type |
|-------|:-----------------:|-----:|---------------|
| Little & Ekvall (1981), model 1 | 3 | 41.13 | per-obs conditional |
| Little & Ekvall (1981), model 2 | 3 | 31.17 | per-obs conditional |
| **Normal + NPMLE (rfl_profile.py)** | **6** | **33.87** | **posterior-weighted $\sum_k \tau_{ik}\hat\mu_k$** |
| **SEV + NPMLE (rfl_sev.py)** | **6** | **28.99** | **posterior-weighted $\sum_k \tau_{ik}\hat\mu_k$** |
| Spindel & Haibach (1981) | 6 | 17.35 | per-obs conditional |
| Bastenaire (1972) | 5 | 20.52 | per-obs conditional |
| Castillo et al. (1985) | 4 | 20.27 | per-obs conditional |
| Castillo & Hadi (1995) | 5 | 18.12 | per-obs conditional |
| Pascual & Meeker (1999), Nor–Nor | 5 | 12.84 | per-obs conditional |
| **Chiu (2005), Nor–Nor (thesis)** | **5** | **10.80** | per-obs conditional |
| **SEV + MCEM (rfl_mcem.py)** | **5** | **10.89** | **per-obs posterior mean $E[Y_i\|\Delta_i, y_i]$** |
| **Normal + INLA-style (rfl_inla.py)** | **5** | **8.63** | **per-obs posterior mean** |
| **SEV + INLA (rfl_sev_inla.py)** ⭐ | **5** | **5.76** | **per-obs posterior mean $E[Y_i\|y_i]=E[\mu_i\|y_i]-\sigma\gamma$** |

> ASSE for "This work" = MAE × 75 = 0.1150 × 75 = **8.625**.
>
> **Why NPMLE methods rank poorly on ASSE despite better statistical inference:**  
> The NPMLE discrete approximation (K=2 atoms) can only assign each specimen to one of two fatigue-limit values {Δ₁, Δ₂}. The posterior-weighted prediction $\sum_k \tau_{ik}\hat\mu_k$ is still constrained to a convex combination of two fixed points, failing to capture individual specimen heterogeneity. All per-obs conditional methods (P&M, INLA, thesis) instead find a continuous $\hat\Delta_i$ per specimen from its own data, achieving far lower ASSE.
>
> **SEV MCEM closes the gap**: MCEM samples $M=200$ values of $\Delta_i$ from the per-observation posterior $P(\Delta_i \mid y_i, S_i)$ via rejection sampling with LogNormal proposal. The fitted value is the SEV posterior predictive mean $E[Y_i \mid y_i] = E[\mu(S_i,\Delta_i)\mid y_i] - \sigma\gamma$ (Euler–Mascheroni correction, $\gamma \approx 0.5772$). This nearly matches Chiu (ASSE=10.80) while preserving the weakest-link SEV physics.
>
> **Why SEV+INLA (5.76) beats Normal+INLA (8.63) despite identical log-likelihoods (−72.88 vs −72.87):**  
> The two models fit the data equally well by likelihood, but differ in how precisely they identify each specimen's latent fatigue limit $\Delta_i$.  
>
> The per-observation posterior is $P(\Delta_i \mid y_i) \propto f(y_i \mid \Delta_i, \sigma) \times g(\Delta_i)$. A **smaller $\sigma$ sharpens the likelihood** as a function of $\Delta_i$, concentrating the posterior more tightly around its mode. SEV achieves the same marginal log-likelihood as Normal but with $\hat\sigma = 0.190$ vs $0.295$. This 36% narrower scale makes each $P(\Delta_i \mid y_i)$ substantially more concentrated, so the posterior-mean fitted value $E[\mu(S_i,\Delta_i)\mid y_i] - \sigma\gamma$ is much closer to $y_i$, cutting ASSE from 8.63 to 5.76.  
>
> Why does SEV find a smaller $\sigma$? The SEV (Gumbel-min) distribution is **left-skewed**, matching the natural asymmetry of log-lifetime data. It can account for the same observed scatter with a tighter scale, whereas the Normal requires a wider $\sigma$ to cover both tails symmetrically. Both models explain the data equally well by marginal likelihood; SEV just does so more efficiently, leaving less posterior uncertainty about each $\Delta_i$.
>
> **The two goals are in tension:**  
> - ASSE ↓ → use continuous per-observation $\hat\Delta_i$ (INLA/MCEM best)  
> - SE calibration ↑ → use NPMLE + Profile Likelihood (corrects Louis 7.7× underestimate)  
>
> A K-larger NPMLE (K → n) would approach continuous-Δ ASSE but lose the parsimony and SE correction properties.

### Evolution of Methods

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
rfl_inla.py             — Normal + INLA: 9-pt Gauss-Hermite + dual-annealing, ASSE = 8.63
          ↓
rfl_sev_inla.py ⭐      — SEV + INLA: SEV conditional + 9-pt GH + dual-annealing
                          ASSE = 5.76 (best result, -33% vs Normal+INLA)
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
                          a=23.2, ASSE=5.74, log-lik=-72.884 (AIC=157.77 vs SEV+INLA 155.76)
                          Conclusion: SEV+INLA (5 params) still preferred by AIC
          ↓
rfl_burr_em.py          — Burr XII + EM-GMM: K-component Gaussian mixture prior on Delta_i,
                          trapezoidal grid E-step (N=400 pts), soft assignments, L-BFGS-B M-step
                          Mode A (sig>=0.15 constraint, K=1): sig=0.160, a=1.6, ASSE=4.09 (best)
                          AIC=157.73 vs SEV+INLA 155.76 (SEV+INLA preferred by AIC)
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

### Why SSLA Instead of Bootstrap?

| Criterion | Bootstrap ($R=500$) | SSLA/ASSLA |
|-----------|--------------------|-----------:|
| Computation | $500 \times$ EM time | $1$–$4 \times$ EM time |
| Randomness | Stochastic (varies by seed) | Deterministic |
| Full 5D SE | Yes (via B resamples) | Yes (via Hessian / refit) |
| Implementation | External loop | Single function call |

Reference: Rodemann J., Marquard A., Augustin T., Caprio M. (2026). *Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification*. TMLR. arXiv:2605.12208.

---

## Burr XII Closed-Form Marginal MLE (`rfl_burr.py`, `rfl_burr2.py`)

These two scripts explore whether **the Gamma-conjugate closed form** can replace numerical integration entirely, yielding a fully closed-form MLE without any quadrature.

### Theoretical background

The SEV density rewrites as:

$$f(y_i \mid \Delta_i) = \frac{c_i}{\sigma} \cdot V_i \cdot e^{-c_i V_i}, \qquad V_i = (S_i - \Delta_i)^{-\beta_1/\sigma},\quad c_i = e^{(y_i - \beta_0)/\sigma}$$

This is a **Gamma(2, $c_i$) kernel** in $V_i$. Placing a Gamma$(\alpha_0, b)$ prior on $V_i$ and integrating out gives the **Burr Type XII** marginal:

$$L_i = \frac{\alpha_0 \, c_i \, b^{\alpha_0}}{\sigma\,(c_i + b)^{\alpha_0 + 1}}$$

No numerical integration — fully closed form.

### Version 1: `rfl_burr.py` — 4-parameter model (degenerate)

**Problem**: the rate $b$ is a single constant shared across all observations. Once $V_i$ is integrated out with a fixed-rate Gamma prior, $\beta_1$ and $S_i$ are **absorbed into $b$** and become unidentifiable. The 4-parameter model $(b_0, \sigma, \alpha_0, b)$ has no access to stress-level structure.

**Consequence**: the MLE drives $b \to 0$, making the posterior mean of $V_i$ degenerate and the fitted value $\hat{y}_i \approx y_i - \text{const}$. ASSE = 3.59 is artifactual — it's a trivial memorisation solution, not a predictive model.

**The fix required**: restore $\beta_1$ and $S_i$ by using a **stress-dependent prior rate**.

### Version 2: `rfl_burr2.py` — 5-parameter model (stress-dependent prior)

**Key idea**: set the prior rate to $b_{S_i} = a / \delta_i^\alpha$ where $\delta_i = S_i - \mu_\Delta$ and $\alpha = -\beta_1/\sigma$. This centres the Gamma prior at $E[V_i] = \delta_i^\alpha = (S_i - \mu_\Delta)^\alpha$ — the expected value of $V_i$ at the prior mean fatigue limit.

This substitution yields a clean closed-form marginal with **all 5 parameters appearing**:

$$\boxed{L_i = \frac{a^{a+1}}{\sigma} \cdot \frac{e^{w_i}}{(a + e^{w_i})^{a+1}}}$$

where:
$$w_i = \frac{y_i - \mu_i^*}{\sigma}, \qquad \mu_i^* = \beta_0 + \beta_1 \log(S_i - \mu_\Delta)$$

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

### MLE Results (`rfl_burr2.py` on P&M 1999 data)

```
b0     = -9.340   b1     = -8.321
sigma  =  0.327   a      =  0.638
mu_d   =  0.528   log-lik = -80.11

ASSE = 15.92   MAE = 0.212   (SEV+INLA: 5.76)
```

Score checks: all 5 equations satisfied to $<10^{-9}$ — the MLE was found correctly.

### Why Burr XII MLE underperforms SEV+INLA

| | SEV+INLA | Burr XII v2 |
|---|---|---|
| Model for $\Delta_i$ | LogNormal$(\mu_d, \sigma_d^2)$ — individual distribution | All $\Delta_i$ share single point $\mu_\Delta$, variability in $V_i$ | 
| Parameters | 5: $\beta_0, \beta_1, \sigma, \mu_d, \sigma_d$ | 5: $\beta_0, \beta_1, \sigma, a, \mu_\Delta$ |
| log-lik | **−72.88** | −80.11 |
| Individual $\Delta_i$ posterior | Per-observation GH integral over LogNormal | Shared via $a$ shape parameter |
| ASSE | **5.76** | 15.92 |

The closed-form gain (no quadrature) comes at the cost of losing **between-individual heterogeneity** in $\Delta_i$: the stress-dependent prior centres all observations at the same $\mu_\Delta$, with the Gamma shape $a$ capturing only collective variability. SEV+INLA's $\sigma_d$ (LogNormal scale) is a genuine per-specimen variance term that SEV+INLA can integrate over — Burr XII has no equivalent.

**Conclusion**: the Burr XII closed-form marginal MLE is theoretically clean and computationally fast (no inner loop), but it is a structurally weaker model than SEV+INLA. The 7.23-unit log-likelihood gap confirms a genuine model misspecification, not just an approximation error.

### Version 3: `rfl_burr_inla.py` — 3-level hierarchy + INLA (the right fix)

**Key idea** (Roy's suggestion): instead of using the Burr XII as the *complete* marginal (Δ already integrated out), use it as the *conditional* likelihood given Δ_i, and let INLA integrate Δ_i over its LogNormal prior.

**3-level hierarchy**:
1. $\Delta_i \sim \text{LogNormal}(\mu_d, \sigma_d^2)$ — INLA outer integral
2. $U_i | \Delta_i \sim \text{Gamma}(a,\; a/(S_i-\Delta_i)^\alpha)$ — Gamma conjugate, closed form
3. $Y_i | U_i \sim \text{SEV}(b_0 - \sigma\log U_i,\; \sigma)$ — conditional SEV

Marginalizing (2)+(3) gives the Burr XII conditional density:
$$f_\text{Burr}(y_i \mid \Delta_i) = \frac{a^{a+1}}{\sigma}\cdot\frac{e^{w_i(\Delta_i)}}{(a + e^{w_i(\Delta_i)})^{a+1}}$$

INLA then integrates over $\Delta_i$. The model has 6 parameters: $(\beta_0, \beta_1, \sigma, a, \mu_d, \sigma_d)$.

**Critical mathematical property**: as $a \to \infty$, $f_\text{Burr}(y_i | \Delta_i) \to f_\text{SEV}(y_i | \Delta_i)$, so this model **nests `rfl_sev_inla.py` exactly**.

**Results**:

```
b0=-9.316  b1=-8.522  sig=0.189  a=23.20  mu_d=-0.644  sig_d=0.035
log-lik = -72.884   (SEV+INLA: -72.880)
ASSE    =   5.74    (SEV+INLA:   5.76)
```

**The a=23.2 finding**: in `rfl_burr2.py` (no per-obs Δ_i), the MLE drove $a=0.638$ — very heavy-tailed Gamma to compensate for the missing individual heterogeneity. Once INLA restores per-observation Δ_i posteriors, only mild overdispersion is needed ($a=23.2$). The two sources of variability are **complementary**, not additive.

**AIC comparison**:

| Model | log-lik | params | AIC |
|-------|:-------:|:------:|:---:|
| SEV+INLA (`rfl_sev_inla.py`) | −72.880 | 5 | **155.76** |
| Burr XII+INLA (`rfl_burr_inla.py`) | −72.884 | 6 | 157.77 |

AIC slightly favours SEV+INLA. The 0.004-nats log-likelihood gain from the extra $a$ parameter is negligible. **`rfl_sev_inla.py` remains the recommended method** (ASSE=5.76, AIC=155.76, 5 params).

---

### Version 4: `rfl_burr_em.py` — Burr XII + EM-GMM Prior

**Key idea**: Instead of a parametric LogNormal prior on $\Delta_i$ (SEV+INLA / Burr+INLA), use a **$K$-component Gaussian mixture** and learn it jointly with the likelihood parameters via EM. This bridges the NPMLE approach (data-driven discrete mixing) with the Burr XII closed-form inner likelihood.

**4-level generative model**:

1. $\Delta_i \sim \sum_{k=1}^K \pi_k\, \mathcal{N}(\mu_k, \sigma_k^2)$ — GMM prior on fatigue limit, EM-estimated
2. $U_i \mid \Delta_i \sim \mathrm{Gamma}\!\bigl(a,\; a/(S_i-\Delta_i)^\alpha\bigr)$ — Gamma conjugate (closed form)
3. $Y_i \mid U_i \sim \mathrm{SEV}(b_0 - \sigma \log U_i,\; \sigma)$ — conditional SEV

Marginalising out $U_i$ in layers (2)+(3) gives the **Burr XII conditional likelihood**:

$$f_\text{Burr}(y_i \mid \Delta_i) = \frac{a^{a+1}}{\sigma}\cdot\frac{e^{w_i(\Delta_i)}}{\bigl(a + e^{w_i(\Delta_i)}\bigr)^{a+1}}, \quad w_i(\Delta_i) = \frac{y_i - b_0 - b_1\log(S_i-\Delta_i)}{\sigma}$$

The remaining integral over $\Delta_i$ is handled by a **trapezoidal grid** (400 uniform points on $\Delta \in [0.002, 0.670]$), not Gauss–Hermite.

#### EM Algorithm

**E-step** — for each observation $i$ and GMM component $k$:

$$\log L_{ik} = \log \int f_\text{Burr}(y_i \mid \delta)\, \mathcal{N}(\delta;\mu_k,\sigma_k^2)\, d\delta \quad \text{(trapezoidal)}$$

$$r_{ik} = \frac{\pi_k\, L_{ik}}{\sum_{k'} \pi_{k'} L_{ik'}}, \quad E[\Delta_i \mid Y_i, Z_i=k] = \frac{\sum_j w_j^{(k)} \delta_j}{\sum_j w_j^{(k)}}$$

where $w_j^{(k)} \propto f_\text{Burr}(y_i \mid \delta_j)\, \mathcal{N}(\delta_j;\mu_k,\sigma_k^2)$ are the per-grid-point weights.

**M-step GMM** — closed form given soft assignments $r_{ik}$ and posterior moments:

$$\pi_k^{\text{new}} = \frac{1}{n}\sum_i r_{ik}, \qquad \mu_k^{\text{new}} = \frac{\sum_i r_{ik}\, E[\Delta_i \mid k]}{\sum_i r_{ik}}, \qquad \sigma_k^{2,\text{new}} = \frac{\sum_i r_{ik}\, E[\Delta_i^2 \mid k]}{\sum_i r_{ik}} - (\mu_k^{\text{new}})^2$$

**M-step likelihood** — L-BFGS-B on the mixture marginal log-likelihood:

$$\ell(b_0, b_1, \sigma, a) = \sum_i \log\sum_k \pi_k L_{ik}$$

#### The Degenerate Solution Problem

Without constraints, the unconstrained MLE drives $\sigma \to 0$ (lower bound 0.05) and $a \to \infty$ (upper bound 200), giving ASSE = 0.49 — apparent **overfitting via degenerate Burr XII**:

As $\sigma \to 0$ and $a \to \infty$, the Burr XII density $f_\text{Burr}(y_i \mid \delta)$ concentrates near a single point $\delta_i^* = S_i - e^{(y_i - b_0)/b_1}$ (the implied fatigue limit for each observation). The GMM prior then memorises all $\delta_i^*$ values, making $\hat{y}_i \approx y_i$. Unlike Gauss–Hermite (which has curvature bias away from the mode), the trapezoidal grid accurately captures this sharp peak — enabling the degenerate solution to be found.

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

| Model | log-lik | params | AIC | ASSE |
|-------|:-------:|:------:|:---:|-----:|
| SEV+INLA (`rfl_sev_inla.py`) | −72.880 | 5 | **155.76** | 5.76 |
| Burr XII+INLA (`rfl_burr_inla.py`) | −72.884 | 6 | 157.77 | 5.74 |
| Burr XII+EM-GMM Mode A K=1 (`rfl_burr_em.py`) | ≈−72.87 | 6 | 157.73 | **4.09** |

**Interpretation**: Mode A achieves the best ASSE by finding a sharper per-observation posterior for $\Delta_i$ (small $\sigma = 0.160$, very mild Gamma prior $a=1.6$), but the AIC still slightly favours SEV+INLA. The $\sigma \geq 0.15$ constraint is heuristic — a formal regularisation framework (e.g., penalised likelihood) would be needed to justify this bound from first principles.

---

## File Structure

```
rfl-inla/
├── rfl_inla.py              # Normal + INLA-style Multiple Laplace (ASSE=8.63)
├── rfl_sev_inla.py          # SEV + INLA: SEV conditional + 9-pt GH (ASSE=5.76) ⭐
├── rfl_profile.py           # Semi-parametric EM + Profile SE (Normal + NPMLE)
├── rfl_sev.py               # SEV + NPMLE: Weibull conditional, BFGS M-step, Profile SE
├── rfl_sev_ksel.py          # K=1..4 selection: Normal+NPMLE vs SEV+NPMLE
├── rfl_mcem.py              # SEV + MCEM: LogNormal g(Delta), rejection sampling E-step
├── rfl_em.py                # Basic EM + BIC model selection
├── rfl_residuals_run.py     # Compute absolute residuals from fitted model
├── ssla_se.py               # SSLA-based SE: Direction A (ASSLA-EM) + B (SSLA-INLA)
├── rfl_burr.py              # Burr XII MLE v1 (4-param, degenerate b->0)
├── rfl_burr2.py             # Burr XII MLE v2 (5-param, stress-dependent prior, ASSE=15.92)
├── rfl_burr_inla.py         # Burr XII + INLA (6-param, 3-level hierarchy, ASSE=5.74)
├── rfl_burr_em.py           # Burr XII + EM-GMM prior (K-comp. Gaussian mix., ASSE=4.09) ⭐
├── data/
│   └── pascual_meeker_1999.csv   # n=75 fatigue dataset (log-life + stress)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install numpy scipy

# 1. Normal + NPMLE (fast, closed-form M-step)
python rfl_profile.py

# 2. SEV + NPMLE (Weibull conditional, BFGS M-step, Profile SE)
python rfl_sev.py

# 3. K=1..4 selection: Normal vs SEV NPMLE
python rfl_sev_ksel.py

# 4. SEV + MCEM (continuous LogNormal g, rejection sampling, ASSE ≈ Chiu thesis)
python rfl_mcem.py

# 5. Normal + INLA-style Multiple Laplace (ASSE=8.63)
python rfl_inla.py

# 6. SEV + INLA: best result, ASSE=5.76
python rfl_sev_inla.py

# 7. Residual analysis
python rfl_residuals_run.py

# 8. SSLA-based SE estimation (Direction A: ASSLA-EM + Direction B: SSLA-INLA)
python ssla_se.py

# 9. Burr XII closed-form MLE v1 (degenerate — b->0, ASSE=3.59 artifactual)
python rfl_burr.py

# 10. Burr XII closed-form MLE v2 (stress-dependent prior, 5-param, ASSE=15.92)
python rfl_burr2.py

# 11. Burr XII + INLA: 3-level hierarchy, nested in SEV+INLA (a->inf), ASSE=5.74
python rfl_burr_inla.py

# 12. Burr XII + EM-GMM: K-component Gaussian mix. prior, best ASSE=4.09 (Mode A, K=1)
python rfl_burr_em.py
```

---

## Conclusions

### Summary of Results

Seven estimation strategies were developed and evaluated on the Pascual & Meeker (1999) fatigue dataset ($n=75$, five stress levels). The key metric is ASSE (Sum of Absolute Errors, lower is better). Three censoring scenarios are also compared — see Cross-Method Censoring Comparison above for full details.

| Method | $f(Y\|\Delta)$ | $g(\Delta)$ | Inner integral | Parameters | ASSE |
|--------|:--------------:|:-----------:|:--------------:|:----------:|-----:|
| Normal + NPMLE (`rfl_profile.py`) | Normal | Discrete (K=2) | Exact sum | 6 | 33.87 |
| SEV + NPMLE (`rfl_sev.py`) | SEV | Discrete (K=2) | Exact sum | 6 | 28.99 |
| **Burr XII v2** (`rfl_burr2.py`) | **SEV** | **Gamma (stress-dep.)** | **Closed form** | **5** | **15.92** |
| SEV + MCEM (`rfl_mcem.py`) | SEV | LogNormal | Monte Carlo (M=200) | 5 | 10.89 |
| Chiu (2005) thesis | Normal | Point mass | Analytic | 5 | 10.80 |
| Normal + INLA (`rfl_inla.py`) | Normal | LogNormal | 9-pt GH | 5 | 8.63 |
| **SEV + INLA** (`rfl_sev_inla.py`) | **SEV** | **LogNormal** | **9-pt GH** | **5** | **5.76** |
| Burr XII + INLA (`rfl_burr_inla.py`) | SEV | LogNormal + Gamma | Burr XII inner + 9-pt GH | 6 | 5.74 |
| **Burr XII + EM-GMM Mode A** (`rfl_burr_em.py`) ⭐ | **SEV** | **GMM (K=1 Gaussian)** | **Burr XII inner + trap. grid** | **6** | **4.09** |

---

### Theoretical Comparison of Methods

#### 1. Discrete vs Continuous Integration over $\Delta$

The fundamental driver of ASSE is **how finely the latent $\Delta_i$ is estimated per observation**.

- **NPMLE** (K=2 atoms): each specimen is assigned to one of two fixed support points $\{\hat\Delta_1, \hat\Delta_2\}$. The posterior-weighted fitted value $\sum_k \tau_{ik}\hat\mu_k$ is a convex combination of only two candidates, which cannot capture individual heterogeneity in $\Delta_i$. This is the reason NPMLE methods have ASSE $\geq 29$ despite excellent inferential properties.

- **MCEM / INLA**: both integrate $\Delta_i$ out over a continuous LogNormal prior, producing a per-observation posterior $P(\Delta_i \mid y_i, S_i)$. The fitted value $E[\mu(S_i, \Delta_i) \mid y_i]$ (with the SEV correction $-\sigma\gamma$) adapts to each specimen's observed life, collapsing ASSE to the single-digit range.

The key inequality: $\text{ASSE}_{\text{discrete-}K} \gg \text{ASSE}_{\text{continuous}}$ for small $K$, regardless of whether Normal or SEV is used.

#### 2. Normal vs SEV Conditional: Why SEV Wins on ASSE

Both Normal+INLA and SEV+INLA achieve **essentially the same marginal log-likelihood** ($-72.87$ vs $-72.88$), confirming neither model is statistically distinguishable on this dataset. Yet ASSE differs by 33% (8.63 vs 5.76). The mechanism is:

**Step 1 — $\sigma$ estimation**: SEV (Gumbel-min) is left-skewed and naturally matches the asymmetry of log-lifetime data. It achieves the same marginal fit with $\hat\sigma = 0.190$, whereas Normal requires $\hat\sigma = 0.295$ to cover both symmetric tails (36% larger).

**Step 2 — Posterior concentration**: the per-observation posterior $P(\Delta_i \mid y_i) \propto f(y_i \mid \Delta_i, \sigma) \times g(\Delta_i)$ is sharpened by a smaller $\sigma$: the likelihood factor is more peaked as a function of $\Delta_i$. With $\sigma=0.190$ (SEV) the posterior concentrates much more tightly around the specimen-specific $\hat\Delta_i$ than with $\sigma=0.295$ (Normal).

**Step 3 — Fitted value precision**: the GH posterior mean $E[\mu(S_i,\Delta_i)\mid y_i] - \sigma\gamma$ is therefore closer to the true generating value, producing smaller $|y_i - \hat{y}_i|$ for every stress level.

This explains the paradox of **equal likelihood but unequal ASSE**: two models can fit the marginal distribution of the data identically while differing substantially in their ability to recover the latent $\Delta_i$ of each specimen.

#### 3. SEV Mean Correction ($-\sigma\gamma$)

Unlike Normal where $E[Y \mid \mu, \sigma] = \mu$, SEV has $E[Y \mid \mu, \sigma] = \mu - \sigma\gamma$ ($\gamma = 0.5772\ldots$ Euler–Mascheroni constant). Using $E[\mu \mid y_i]$ directly as the fitted value introduces a systematic positive bias of $\sigma\gamma \approx 0.164$ (at $\sigma \approx 0.285$). This mistake accounts for ASSE rising from $\approx 11$ to $\approx 15$ in the uncorrected MCEM runs. The corrected estimator $E[\mu \mid y_i] - \sigma\gamma$ is the true posterior predictive mean.

#### 4. GH Quadrature vs Monte Carlo E-step

Both MCEM and SEV+INLA use a continuous LogNormal $g(\Delta)$, but differ in how they approximate the posterior integral:

| Criterion | MCEM (M=200) | 9-pt GH (INLA) |
|-----------|:---:|:---:|
| Deterministic | No (MC noise ±1–2 in LL) | Yes |
| Iterations to converge | 70–80 | 1 (direct MLE) |
| ASSE | 10.89 | 5.76 |
| LL at convergence | −12.5 (MC approx) | −72.88 (exact) |

The 9-pt GH evaluates the integrand at optimal quadrature nodes centred at the per-observation posterior mode, achieving far higher accuracy than 200 random samples. The MC noise in MCEM also pollutes the M-step BFGS, preventing convergence to the true MLE; this partly explains the 1.88 log-unit gap between MCEM ($\ell \approx -74.76$) and SEV+INLA ($\ell = -72.88$).

---

### Pros and Cons

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **NPMLE + Profile SE** (`rfl_profile.py`) | Gold-standard SE calibration; corrects Louis 7.7× underestimate; no parametric $g$ assumption | High ASSE; discrete $\Delta$ too coarse for prediction |
| **SEV + NPMLE** (`rfl_sev.py`) | Physical motivation (weakest-link); better AIC/BIC than Normal+NPMLE | Same ASSE limitation as Normal+NPMLE |
| **SEV + MCEM** (`rfl_mcem.py`) | Continuous $g(\Delta)$; good warm-start for INLA; flexible | MC noise; slow convergence; needs M≥200, iter≥70 |
| **Normal + INLA** (`rfl_inla.py`) | Accurate GH integration; fast NM/SA convergence | Symmetric Normal misspecifies left-skewed log-life; ASSE 8.63 |
| **SEV + INLA** (`rfl_sev_inla.py`) | Best ASSE (5.76) among GH methods; physically correct conditional; deterministic | Degenerates at 37.3% censoring ($\hat\beta_1\to-20$, same as Normal+INLA); SE not yet Profile-calibrated |
| **Burr XII + EM-GMM** (`rfl_burr_em.py`) | Best ASSE (4.09); **robust under heavy censoring** ([C] ASSE=4.43 vs SEV+INLA 17.97); no Laplace curvature bias | Degenerate MLE without $\sigma,a$ constraints; heuristic bounds; AIC still favours SEV+INLA |

---

### Limitations

1. **In-sample ASSE**: all ASSE figures are computed on the same data used for fitting. Out-of-sample cross-validation (leave-one-out or $k$-fold) would give more conservative estimates and could change the relative rankings, particularly for methods with concentrated posteriors (SEV+INLA).

2. **Small sample ($n=75$)**: with only 15 specimens per stress level, parameter estimates and ASSE rankings may not generalise to other fatigue datasets. The near-degenerate $g(\Delta)$ ($\hat\sigma_\Delta \approx 0.03$) in particular may be an artefact of small $n$.

3. **Unimodal LogNormal $g(\Delta)$**: NPMLE analysis suggests the mixing distribution may have more than one support point (e.g., K=3 has lowest BIC for SEV). The LogNormal assumption, while tractable, cannot represent multimodal distributions of the fatigue limit.

4. **SE for SEV+INLA not Profile-calibrated**: standard errors for the SEV+INLA model are available only via numerical Hessian (3×3, nuisance fixed) or ASSLA approximation — not profile likelihood. The 7.7× Louis underestimation documented for Normal+NPMLE may persist here.

5. **Censoring robustness**: simulated censoring experiments (see Cross-Method Censoring Comparison above) show that GH-based methods (Normal+INLA, SEV+INLA) both degenerate at 37.3% hybrid censoring ($\hat\beta_1 \approx -20$). The Burr+EM-GMM trapezoidal grid remains stable across all three censoring levels but requires explicit $a \geq 1$ regularisation to prevent $S_\text{Burr}\to 1$ degeneracy. The ASSE ranking under heavy censoring reverses significantly relative to the uncensored case.

6. **Convergence of MCEM**: the MC log-likelihood oscillates even at M=200; convergence is assessed by parameter stability rather than the standard EM monotone likelihood criterion. A Rao-Blackwellised estimator or larger M would be needed for formal convergence guarantees.

---

### Future Directions

- **EM-GMM regularisation**: the Mode A ($\sigma \geq 0.15$) constraint in `rfl_burr_em.py` is heuristic. A principled approach — penalised likelihood, minimum-description-length, or posterior predictive LOO validation — is needed to justify the constraint and confirm ASSE=4.09 generalises out-of-sample.
- **Profile SE for SEV+INLA**: extend the `rfl_profile.py` Profile Likelihood approach to the 5D SEV+INLA parametrisation for calibrated confidence intervals.
- **Leave-one-out ASSE**: implement LOO cross-validation to obtain bias-corrected ASSE estimates, using importance sampling on the per-observation posteriors.
- **Censoring regularisation for EM-GMM**: the $a \geq 1$ constraint in `rfl_burr_em.py` is heuristic. Under [B][C], $a$ binds at 1.0 — a proper Gamma hyperprior on $a$ (e.g., $a \sim \text{Gamma}(\alpha_0, \beta_0)$ with $\alpha_0 > 1$) would provide principled regularisation and allow full Bayesian uncertainty quantification on $a$.
- **Bayesian posterior via MCMC**: replace the MLE outer optimisation with a full Bayesian treatment of $\theta = (\beta_0, \beta_1, \sigma, \mu_\Delta, \sigma_\Delta)$, using the SEV+GH likelihood in a Metropolis–Hastings sampler.

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

### Reliability Data & Extreme Value Distributions

8. **Meeker, W. Q. and Escobar, L. A.** (1998). *Statistical Methods for Reliability Data*. New York: John Wiley & Sons.
    > Standard reference for Weibull/SEV lifetime models, accelerated life testing, and the physical basis of the smallest extreme value distribution in reliability engineering.

9. **Fisher, R. A. and Tippett, L. H. C.** (1928). "Limiting forms of the frequency distribution of the largest or smallest member of a sample." *Proceedings of the Cambridge Philosophical Society*, **24**(2), 180–190.
    > Theoretical foundation of extreme value theory: the limiting distribution of the minimum of many i.i.d. random variables is the Gumbel-min (SEV) distribution — the weakest-link justification for the SEV conditional in the RFL model.

### Mixture Models & Semi-parametric Theory

10. **Lindsay, B. G.** (1983). "The geometry of mixture likelihoods: A general theory." *Annals of Statistics*, **11**(1), 86–94.

11. **Murphy, S. A. and van der Vaart, A. W.** (2000). "On profile likelihood." *Journal of the American Statistical Association*, **95**(450), 449–465.

12. **Teicher, H.** (1963). "Identifiability of finite mixtures." *Annals of Mathematical Statistics*, **34**(4), 1265–1270.

### INLA — Integrated Nested Laplace Approximation

13. **Rue, H., Martino, S., and Chopin, N.** (2009). "Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations." *Journal of the Royal Statistical Society, Series B*, **71**(2), 319–392.
    > Two-level Laplace framework adopted here: inner per-latent-variable approximation + outer hyperparameter search.

### Monte Carlo EM

14. **Wei, G. C. G. and Tanner, M. A.** (1990). "A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms." *Journal of the American Statistical Association*, **85**(411), 699–704.
    > Original MCEM paper: replaces the intractable E-step expectation with a Monte Carlo average over samples from the complete-data posterior. Forms the theoretical basis of `rfl_mcem.py`.

### Heuristic Learning — Simulated Annealing & Dual Annealing

15. **Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P.** (1983). "Optimization by simulated annealing." *Science*, **220**(4598), 671–680. DOI: 10.1126/science.220.4598.671
    > Original SA: temperature-adaptive random walk for global optimisation of non-convex objectives.

16. **Tsallis, C. and Stariolo, D. A.** (1996). "Generalized simulated annealing." *Physica A*, **233**, 395–406.
    > Theoretical basis of dual annealing: Tsallis statistics replace the Boltzmann acceptance criterion.

17. **Xiang, Y., Sun, D. Y., Fan, W., and Gong, X. G.** (1997). "Generalized simulated annealing algorithm and its application to the Thomson model." *Physics Letters A*, **233**, 216–220.
    > Practical GSA algorithm; forms the core of `scipy.optimize.dual_annealing`.

18. **Xiang, Y. and Gong, X. G.** (2000). "Efficiency of generalized simulated annealing." *Physical Review E*, **62**, 4473.
    > Efficiency analysis of GSA; used by SciPy's dual-annealing implementation.

### Local Optimisation

19. **Nelder, J. A. and Mead, R.** (1965). "A simplex method for function minimization." *The Computer Journal*, **7**(4), 308–313.
    > Derivative-free downhill simplex (Stage 3 polish in optimisation pipeline).

### Error-in-Variables & Measurement Error

20. **Fuller, W. A.** (1987). *Measurement Error Models*. New York: John Wiley & Sons.

### Self-Supervised Uncertainty Quantification

21. **Rodemann, J., Marquard, A., Augustin, T., and Caprio, M.** (2026). "Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification." *Transactions on Machine Learning Research*. arXiv:2605.12208.
    > SSLA/ASSLA: bypass parameter posterior, directly approximate posterior predictive by refitting on self-predicted data. Deterministic, sampling-free. Applied in `ssla_se.py` to fix the Louis 7.7× underestimation and provide full 5D UQ.

### Thesis

22. **Chiu, C.-H.** (2005). *A Family of Bivariate Distributions With Some Applications to Statistical Inferences*. M.Sc. thesis, Graduate Institute of Management Sciences, Tamkang University (Chapter 2: Random Fatigue-Limit Model via error-in-measurement regression, ASSE = 10.80).

---

## Licence

MIT
