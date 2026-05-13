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

| Stress $S$ | $n_j$ | MAE | Max $|e_i|$ |
|------------|-------|-----|-------------|
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
```

---

## File Structure

```
rfl-inla/
├── rfl_inla.py              # INLA-style Multiple Laplace main script
├── rfl_profile.py           # Semi-parametric EM + Profile SE (existing)
├── rfl_em.py                # Basic EM + BIC model selection
├── rfl_residuals_run.py     # Compute absolute residuals from fitted model
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

### Thesis

18. **Chiu, C.-H.** (2005). *A Family of Bivariate Distributions With Some Applications to Statistical Inferences*. M.Sc. thesis, Graduate Institute of Management Sciences, Tamkang University (Chapter 2: Random Fatigue-Limit Model via error-in-measurement regression, ASSE = 10.80).

---

## Licence

MIT
