# Stage 11.1 — Unobserved Components / structural time series

**Prerequisite reading:** `stage-11-overall-handoff.md`, and §5 and §7 of
`stage-11.0-primitives-handoff.md` (partitioned initialisation, and the `_ols`
weight guard — the only two 11.0 pieces this item needs).

**This item does not need the multivariate filter.** Everything here is a
univariate observation, which `TimeVaryingSSM` already handles. That is what
makes it the cheapest item in the stage and the right one to build first.

**Difficulty: Easy–Moderate.** The references amount to roughly 600 lines of
substantive code and **none of it is new numerics** — it is a mapping from a
component specification onto `(Z, T, R, Q, H)` matrices that an existing,
tested filter and smoother then consume.

---

## 1. What this builds

A decomposition of a single series into named, interpretable, *stochastic*
components:

```
y_t = μ_t + γ_t + c_t + ψ_t + x_t'β + ε_t
      level  seas  cycle  AR    regression  irregular
```

where each component has its own variance, estimated by maximum likelihood, and
each is recovered with a standard error by the smoother. The contrast with
Stage 3's decompositions is the point and belongs in the docs: `stl_decompose`
splits a series by a filter you choose; this one splits it by a *model* that is
fitted, so every component comes with an interval and the split is testable.

### Exported surface

| Name | Kind | Notes |
|---|---|---|
| `fit_uc` | function | the fitter |
| `UCModel` | type | `<: StateSpaceModel`, `<: UnivariateModel` |
| `uc_decomposition` | function | → `UCDecomposition`, the plot/inspection payload |
| `UCDecomposition` | type | per-component smoothed path + standard error |
| `forecast` / `predict` | methods | exact aliases, per `src/forecast.jl:434` [U] |

`UnivariateModel`'s docstring at `src/abstract.jl:40` says "as opposed to
VAR/VECM, which are inherently multivariate" [S] — a UC model is univariate, so
it belongs there, and this is the first Stage 11 type that does.

---

## 2. The component → state-space mapping, verified

Every matrix below was read off a constructed `statsmodels` model, not recalled.
[V] `s11/uc_matrices.py`

### 2.1 Local level (random walk)

```
k_states = 1    states: [level]
Z = [1]         T = [1]        R = [1]
params: sigma2.irregular, sigma2.level
```

### 2.2 Local linear trend

```
k_states = 2    states: [level, trend]
Z = [1 0]       T = [1 1; 0 1]      R = I₂
params: sigma2.irregular, sigma2.level, sigma2.trend
```

### 2.3 Dummy seasonal, period s

`s − 1` states. For `s = 4`, combined with a local linear trend:

```
k_states = 5    states: [level, trend, seasonal, seasonal.L1, seasonal.L2]
Z = [1 0 1 0 0]
T = [1  1  0  0  0
     0  1  0  0  0
     0  0 -1 -1 -1
     0  0  1  0  0
     0  0  0  1  0]
R' = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0]      (k_posdef = 3)
params: sigma2.irregular, sigma2.level, sigma2.trend, sigma2.seasonal
```

The `-1 -1 -1` row is the seasonal sum-to-zero constraint. **Note `k_posdef = 3`
against `k_states = 5`** — the lagged seasonal states carry no disturbance, so
`R` is 5×3 and not square. Getting `R` square would silently add variance to the
seasonal lags.

### 2.4 Trigonometric (frequency-domain) seasonal

Two states per harmonic `j`, with `λ_j = 2πj/s` and a rotation block

```
[ cos λ_j   sin λ_j ]
[ -sin λ_j  cos λ_j ]
```

Verified for `s = 4`, `harmonics = 2`: harmonic 1 has `λ = π/2` giving
`[0 1; -1 0]`, harmonic 2 has `λ = π` giving `[-1 0; 0 -1]`, and the observed
`T` is exactly block-diagonal with those two blocks. [V]

```
k_states = 5    states: [level, freq_seasonal.0 … freq_seasonal.3]
Z = [1 1 0 1 0]        # only the cosine state of each harmonic loads
params: sigma2.irregular, sigma2.level, sigma2.freq_seasonal_4(2)
```

**One variance for the whole block**, not one per harmonic — and the parameter
name encodes period and harmonic count: `sigma2.freq_seasonal_4(2)`.

At `harmonics = s/2` the final harmonic's sine state is structurally zero, the
same degeneracy `fourier_terms` already documents from Stage 9B Tier 2.4 (`K =
period/2` drops the sine column because `sin(πt) = 0` for integer `t`) [U].
**Reuse that reasoning and cite it** — the package has already made this
decision once and should not make it differently here.

### 2.5 Damped stochastic cycle

Two states, rotation scaled by the damping factor `ρ`:

```
k_states += 2   states: [cycle, cycle.auxilliary]
T block = ρ · [ cos λ   sin λ ; -sin λ   cos λ ]
Z loads the first state only
params: sigma2.cycle, frequency.cycle, damping.cycle
```

[V] Three extra parameters, two of them **not** variances — so the parameter
transform is not uniformly "exponentiate". `frequency.cycle` needs a positivity
(and sensibly an upper) bound; `damping.cycle` needs `0 < ρ < 1`.

### 2.6 Autoregressive component, order p

A companion block:

```
k_states += p   states: [ar.L1 … ar.Lp]
T block = companion(φ)           # [φ₁ φ₂; 1 0] for p = 2
Z loads ar.L1
R loads ar.L1 only  (k_posdef counts it once)
params: sigma2.ar, ar.L1 … ar.Lp
```

[V] for `p = 2`: `T` block came back `[-0.0772 -0.9185; 1 0]` with
`k_posdef = 2` against `k_states = 3`.

**This is the block that makes §5 of 11.0 necessary** — it is stationary while
everything above it is diffuse.

`companion` from Stage 10.0 builds this [U]; reuse it rather than writing a
second one.

### 2.7 Regression terms

`x_t'β` enters either as a fixed coefficient (absorbed into the observation
equation) or as a state with its own variance (a drifting coefficient). The
drifting case is **already built** — Stage 8.3's `time_varying_regression` path
in `arimax.jl` does exactly this and calls `kalman_filter_diffuse` with
`diffuse_idx = (r_arma+1):(r_arma+k)` [S] `src/arimax.jl:269`.

**Do not rebuild it.** Either reuse that construction or, better, note in the
docs that `fit_arimax(…, model=:tvss)` is the supported route for drifting
coefficients and keep 11.1's regression block fixed-coefficient only. The
charter's §9 principle of not building what exists applies.

---

## 3. Specification API — and the trap to not copy

### 3.1 statsmodels' interface is a trap (charter defect D4)

Measured behaviour of the `level=` argument [V] `s11/uc_trap.py`:

| call | k | stochastic (level, trend) | irregular | parameters |
|---|---|---|---|---|
| `level=True` | 1 | **(False, False)** | True | `sigma2.irregular` only |
| `level='local level'` | 1 | (True, False) | True | irregular, level |
| `level=True, stochastic_level=True` | 1 | (True, False) | **False** | level only |
| `level='dtrend'` | 2 | (False, False) | True | irregular only |
| `level='lltrend'` | 2 | (True, True) | True | irregular, level, trend |
| `level='rwalk'` | 1 | (True, False) | **False** | level only |
| `level='rwdrift'` | 2 | (True, False) | **False** | level only |
| `level='smooth trend'` | 2 | (False, True) | True | irregular, trend |
| `level='random trend'` | 2 | (False, True) | **False** | trend only |

Three things are wrong with this as a design, and all three are avoidable:

1. **`level=True` gives a *deterministic* level.** A boolean that reads as "yes,
   include a level" silently means "include a constant".
2. **The string shortcuts disagree about the irregular.** `'local level'` has
   one, `'rwalk'` does not, and nothing in the name says so.
3. **A specification with no stochastic element emits a `SpecificationWarning`
   and silently adds an irregular** rather than refusing.

### 3.2 What to build instead

An explicit component specification, with the stochastic/deterministic choice
never implied:

```julia
fit_uc(y;
       level        = :none,        # :none | :deterministic | :stochastic
       trend        = :none,        # :none | :deterministic | :stochastic
       seasonal     = nothing,      # nothing | (period, :dummy|:trig, nharmonics)
       seasonal_kind = :dummy,
       cycle        = :none,        # :none | :deterministic | :stochastic
       damped_cycle = false,
       ar           = 0,
       irregular    = true,
       exog         = nothing,
       init_kappa   = 1e6,
       ...)
```

and **convenience constructors for the named models**, which is the genuinely
useful half of statsmodels' string interface:

| name | expands to |
|---|---|
| `uc_local_level(y)` | level `:stochastic`, irregular |
| `uc_local_linear_trend(y)` | level + trend `:stochastic`, irregular |
| `uc_random_walk(y)` | level `:stochastic`, **no irregular** |
| `uc_smooth_trend(y)` | level `:deterministic`, trend `:stochastic`, irregular |
| `uc_bsm(y, period)` | level + trend + seasonal `:stochastic`, irregular — R's `type="BSM"` |

Each convenience constructor's docstring states its exact expansion, so the
mapping is documented rather than inferred.

**Refuse, do not warn.** A specification with every component deterministic and
`irregular=false` has no disturbance at all and is not a model; throw an
`ArgumentError` naming the fact that at least one component must be stochastic or
the irregular must be present. This is the honest-refusal pattern and it is
exactly the case statsmodels papers over.

---

## 4. Initialisation — the decision and why it is not an oversight

**Approximate diffuse with κ = 1e6 on the non-stationary states, stationary on
the AR block.** This is 11.0 §5's `SSMInit`.

The reference does the same: measured on a level + trend + seasonal(4) + damped
cycle + AR(1) model, `k_states = 8`,

```
blocks: {(0,1,2,3,4,5,6): 'approximate_diffuse', (7,): 'stationary'}
approximate_diffuse_variance = 1000000.0
```

[V] `s11/uc_init.py`

**This must be said out loud in the user-facing docstring**, because the package
*has* an exact diffuse filter (`kalman_filter_diffuse`, Stage 8.2) and a reader
who knows that will assume its absence here is an oversight. It is not: the
reference uses approximate diffuse, matching it is what makes the numbers
comparable, and the exact filter has no smoother anyway (11.0 §10).

Expose `init_kappa` so a user can change it, and document that the likelihood
depends on it — it is a prior, not a technicality.

### What R does, for the record

`StructTS` uses `a0 = y[1]` and `P0 = 1e6 · var(y)/100`, i.e. `1e4 · var(y)`
[S] deparsed lines 12, 86–87. On the Nile that is `P0 = 2.86379e8` against
statsmodels' flat `1e6`. It also makes **the first innovation identically zero**
(charter defect D2, measured `resid[1] = 0.000e+00` [V]), discarding one
observation's information. Neither choice is ours; ours is statsmodels'.

---

## 5. Estimation

### 5.1 Parameterisation

Variances by `log` transform, so the optimiser works unconstrained and the
variance is positive by construction. **Not** by a bound, and not by clipping —
charter defect D3 is R clipping with `pmax(coef * vx, 0)`, which makes a
boundary solution indistinguishable from a clipped negative one.

Non-variance parameters:

| parameter | transform |
|---|---|
| `frequency.cycle` | logistic onto `(0, π)` |
| `damping.cycle` | logistic onto `(0, 1)` |
| `ar.L1 … ar.Lp` | `_constrain_stationary` from 11.0 §6 at `k = 1` |

### 5.2 Scaling, and a warning from R

R optimises on variances **scaled by `var(y)/100`** and multiplies back at the
end [S] deparsed lines 86, 94–97, 106–107, 118. That is a sensible
conditioning trick and worth imitating *internally*; it is also where R's
clipping lives, so imitate the scaling and not the clipping.

Starting values: the reference spends **94 lines** on `start_params` [S]
`structural.py:889-982`, mostly branching. A defensible simple rule — split
`var(y)` equally across the stochastic components, with the irregular taking
half — is enough to reach the same optimum on the Nile, and §7.3 is the test
that checks the optimiser is not start-dependent on the well-behaved fixtures.

### 5.3 `_optimize`, and what to watch

`src/optim.jl:57` [S]. Two cautions:

- The objective must be **generic in the parameter eltype** so ForwardDiff's
  `Dual` passes through. That means the `(Z,T,R,Q,H)` builder allocates on
  `promote_type(eltype(params), Float64)`, as `build_statespace` already does
  (`GaussianSSM{S}` is parameterised exactly for this reason, and its docstring
  says so [S]).
- **Report convergence honestly.** `OptimResult` carries the flag; surface it on
  `UCModel` and print it in `show`. A UC likelihood is flat (§7.1) and a user
  needs to know the optimiser stopped rather than converged.

---

## 6. Component extraction

```julia
uc_decomposition(m::UCModel) -> UCDecomposition
```

carrying, for each component present, the smoothed path **and its standard
error**, taken from the existing smoother's `alpha` and `V`:

```julia
struct UCDecomposition
    level::Union{Nothing,Vector{Float64}};      level_se::Union{Nothing,Vector{Float64}}
    trend::Union{Nothing,Vector{Float64}};      trend_se::…
    seasonal::…;  cycle::…;  ar::…;  regression::…
    irregular::Vector{Float64}
    names::Vector{Symbol}        # components actually present, in order
    period::Union{Nothing,Int}
end
```

The standard errors are the thing the reference gives you and Stage 3's
decompositions cannot: `sqrt.(diag(V[t]))` for the relevant state index. That is
the whole argument for this item existing alongside `stl_decompose`.

The reference's component accessors total **353 lines** across seven properties
(`level` 37, `trend` 38, `seasonal` 40, `freq_seasonal` 83, `cycle` 47,
`autoregressive` 46, `regression_coefficients` 61) [S], almost all of it index
bookkeeping. In Julia this is one function over a recorded state-index map built
by the same code that built `Z` — keep the map, don't recompute the offsets.

**Reuse `StateDecomposition` for the plot.** The plot-recipes work already built
it for ETS and TBATS state decompositions, and a UC decomposition is the same
shape. See the plot handoff; the point here is that `UCDecomposition` is the
*fit* payload and `StateDecomposition` is the *plot* payload, and converting
between them is a few lines.

---

## 7. Verification plan

### 7.1 The headline: the references agree to 1.26e-08 once initialisation is matched

This is charter §5.1 and it is the spine of the test suite.

**The reconciliation, as a test.** Build the local-level model, set
`a0 = y[1] = 1120.0`, `P0 = 1e6·var(y, ddof=1)/100 = 2.86379e8`, and fix
`σ²_level = 1469.146619`, `σ²_irregular = 15098.577154`. Then:

| quantity | R `StructTS` / `KalmanRun` | agreement |
|---|---|---|
| filtered state, 100 periods | `[1120.0, 1140.927413, 1072.798595, …]` | **max abs diff 1.257e-08** |
| smoothed state, 100 periods | from `tsSmooth` | **max abs diff 9.969e-09** |

[V] `s11/uc_fixed2.R` exports `uc_R_filter.csv`; `s11/uc_match.py` compares.

**The 1e-08 is R's print precision, not a disagreement** — the variances fed back
in carry 7 significant digits. A Julia test can do better than this by using the
full-precision variances, and should: assert **1e-10** against statsmodels at
fixed parameters, and keep the R comparison as a looser independent check at
1e-07.

### 7.2 Tier-1 targets (fixed parameters, tight)

| fixture | parameters | target | tolerance |
|---|---|---|---|
| Nile, local level | σ²_ε = 15098.577, σ²_η = 1469.147 | llf = **−632.537695** | 1e-9 |
| Nile, local level | σ²_ε = 15108.3155, σ²_η = 1463.5471 | llf = **−632.53768559** | 1e-9 |
| Nile, local level, R's init | same as row 1, `a0 = 1120`, `P0 = 2.86379e8` | llf = **−632.5456059039** | 1e-9 |
| Nile, local level, default init | row 1 params, `a0 = 0`, `P0 = 1e6` | filtered/smoothed states vs `uc_R_filter.csv` | 1e-7 |

[V] all from `s11/uc_diag.py`, `s11/uc_conv.py`, `s11/uc_match.py`

Note rows 1 and 3 differ **only** in initialisation and differ by 0.0079 in
log-likelihood. That pair is the test that catches an initialisation bug, and it
is worth a comment saying so.

### 7.3 Tier-2 targets (fitted optima, loose) — with the flatness measured

The Nile optimum is identified to about **three significant figures**, measured
across four optimisers [V] `s11/uc_conv.py`:

| optimiser | σ²_irregular | σ²_level | llf |
|---|---|---|---|
| L-BFGS (statsmodels default) | 15078.0100 | 1478.8120 | −632.53776058 |
| L-BFGS, tightened | 15108.3145 | 1463.5472 | −632.53768559 |
| Nelder-Mead | 15108.3159 | 1463.5471 | −632.53768559 |
| Powell | 15108.3152 | 1463.5471 | −632.53768559 |
| BFGS | 15108.3157 | 1463.5472 | −632.53768559 |
| R `StructTS` (= Durbin & Koopman published) | 15098.5770 | 1469.1470 | −632.53769527 |

The spread across the four converged optimisers is **1.4e-04 relative**; the
spread including R's point is **3.8e-03 relative** on σ²_level, for a
log-likelihood difference of **9.7e-06**.

**So: assert the fitted σ² to 1e-2 relative and the fitted llf to 1e-5.**
Anything tighter is asserting a stopping rule. State the measured spread in the
testset's comment so the number is defensible rather than arbitrary.

**Also assert that our llf at our optimum is ≥ the llf at R's parameters** — a
one-sided check that our optimiser is not worse than the reference's, which is
the thing actually worth guaranteeing and which the flatness makes testable.

### 7.4 Tier-3: the boundary fixture

`log10(UKgas)`, 108 observations, BSM specification. Charter §5.5:

| | σ²_level | σ²_trend | σ²_seasonal | σ²_irregular |
|---|---|---|---|---|
| R `StructTS(type="BSM")` | **0.000000e+00** | 1.733003e-05 | 7.136943e-04 | 3.677978e-04 |
| statsmodels | 2e-10 | 1.503e-06 | 6.245776e-04 | 3.433252e-04 |

[V] `s11/bsm.py`

The trend differs by **11.5×** and both are correct, because the level sits on a
boundary and the rest are then weakly identified. **Assert only:**

- `σ²_level < 1e-6` (the qualitative finding both references agree on)
- `σ²_seasonal` and `σ²_irregular` within 20% of the statsmodels figures
- the fitted log-likelihood ≥ statsmodels' **172.46517259**
- **no clipping**: our `σ²_level` is strictly positive, not exactly zero — the
  deliberate contrast with R's defect D3, and worth a comment naming it

Do **not** assert the trend variance.

### 7.5 Reductions

1. **Deterministic level + irregular reduces to OLS on a constant.** The fitted
   level must equal `mean(y)` and the irregular variance must equal the OLS
   residual variance, to 1e-8. Cheapest possible structural check.
2. **Local level with σ²_level → 0 reduces to a constant mean.** Fit with the
   level variance fixed at 1e-12 and check the smoothed level is flat to 1e-6.
3. **Local level with σ²_irregular → 0 reduces to the series itself.** The
   smoothed level must equal `y` to 1e-6.
4. **Dummy and trigonometric seasonal with full harmonics span the same space.**
   At `harmonics = s/2` for even `s`, a trigonometric seasonal has the same
   number of free seasonal states as a dummy seasonal (`s − 1`, once the
   degenerate sine state is dropped), so the two fitted log-likelihoods must
   agree to ~1e-6. **This is the test that catches a wrong rotation block**, and
   it needs no reference at all.
5. **An AR(p) component alone, with no level and no irregular, reduces to
   `fit_arma(y, p, 0)`.** The package already has that, already verified. Agree
   to 1e-6 on the log-likelihood.

Reduction 4 and 5 are the two most valuable because they test against things
already in the repository rather than against a reference.

### 7.6 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | local level, llf at three fixed parameter vectors (§7.2) | 10 |
| 2 | filtered/smoothed state against `uc_R_filter.csv` | 14 |
| 3 | initialisation sensitivity: rows 1 vs 3 of §7.2 differ by 0.0079 | 8 |
| 4 | local level fitted optimum, tier-2 tolerances | 10 |
| 5 | our optimum is no worse than R's parameters | 4 |
| 6 | local linear trend, state-space matrices exactly as §2.2 | 10 |
| 7 | dummy seasonal matrices exactly as §2.3, including `k_posdef = 3` | 16 |
| 8 | trigonometric seasonal rotation blocks as §2.4 | 14 |
| 9 | damped cycle matrices and parameter bounds | 12 |
| 10 | AR component companion block, and its stationary init | 14 |
| 11 | partitioned init: diffuse + stationary blocks as §4 | 12 |
| 12 | UKgas boundary fixture, tier-3 assertions only | 12 |
| 13 | no clipping: σ² strictly positive at the boundary | 4 |
| 14 | reduction 1: deterministic level = OLS constant | 8 |
| 15 | reduction 2 and 3: variance limits | 12 |
| 16 | reduction 4: dummy vs trig seasonal llf agreement | 8 |
| 17 | reduction 5: AR-only = `fit_arma` | 8 |
| 18 | spec refusal: no stochastic element and no irregular throws | 10 |
| 19 | spec refusal: seasonal period < 2, harmonics > s/2, ar < 0 | 14 |
| 20 | `uc_decomposition` component paths sum to the fitted values | 10 |
| 21 | component standard errors are positive and finite | 8 |
| 22 | missing data: NaN in `y` is skipped, llf drops accordingly | 12 |
| 23 | `forecast`/`predict` alias identity; forecast variance grows | 12 |
| 24 | `show` output includes convergence status | 6 |
| 25 | performance: allocations in the likelihood, `@code_warntype` | 8 |

**25 testsets, ≥ 276 assertions.**

---

## 8. Forecasting

The state-space forecast is `a_{n+h}` propagated by `T`, with variance
`P_{n+h}` accumulating `R Q R'`, and the observation variance adding `H`. The
existing `Forecast` struct (`src/forecast.jl:15`) fits — univariate point, se,
levels, lower/upper matrices [U] — so **reuse it** rather than adding a type.

Two things to get right:

- **A diffuse state's forecast variance grows without bound**, which is correct
  and should not be suppressed. With κ = 1e6 the intervals start wide; say so in
  the docstring.
- **Deterministic components do not contribute forecast variance.** Assert it:
  a model with a deterministic level and no irregular has zero forecast
  variance, which is a good degenerate test.

---

## 9. Charts

Detail is in the plot-recipes handoff; what 11.1 owes it:

| plot | source | note |
|---|---|---|
| component panels (level, trend, seasonal, cycle, AR, irregular) | reference `plot_components` is **216 lines** [S] | **reuse `StateDecomposition`** — this is the same shape as the ETS state decomposition already built |
| component bands | `V` from the smoother | the reference offers this; `plot.ets` in R does **not**, and R's `plot.StructTS` draws at most the three BSM states |
| fitted vs observed | trivial | |

The one design note: R's `plot.ets` draws **at most four panels** using `s1`
alone for the seasonal [U, from the Stage 9 work], and `plot.StructTS` is
similarly limited. Ours should draw one panel per *present* component, which the
`names` field on `UCDecomposition` makes straightforward.

---

## 10. Budget

| Piece | Reference | Julia estimate |
|---|---|---|
| spec parsing + named constructors | `__init__` 306 lines (≈150 substantive) [S] | 120–160 |
| `(Z,T,R,Q,H)` builder | `setup` **141 lines** [S] | 130–180 |
| parameter packing / transforms / names | `update` 45 + transforms 80 + names 61 [S] | 150–200 |
| starting values | `start_params` 94 [S] | 40–60 |
| `uc_decomposition` | accessors 353 [S] | 80–120 |
| forecast | — | 40 |
| `show` | `summary` 40 [S] | 40 |
| **total** | ≈600 substantive reference lines | **600–800** |

No new numerics. The filter, the smoother, `_optimize`, `companion` and (after
11.0 §5) the partitioned initialisation all exist.

---

## 11. What not to do

1. **Do not build the exact diffuse smoother.** §4. The reference uses
   approximate diffuse; matching it is the goal.
2. **Do not copy the `level=`/string interface.** §3.1 — it has three distinct
   traps and an explicit spec avoids all of them.
3. **Do not clip variances at zero.** Charter defect D3. Fit on the log scale and
   let a boundary solution be a small positive number that a user can see.
4. **Do not compare to `StructTS$loglik`.** Charter defect D1 — it is not a
   log-likelihood, and on the Nile it is off by 10.663 from the true value at its
   own parameters.
5. **Do not assert the UKgas trend variance.** §7.4 — the references differ by
   11.5× and both are right.
6. **Do not rebuild drifting regression coefficients.** §2.7 — Stage 8.3 has
   them.
7. **Do not warn-and-patch an unidentified specification.** Refuse it.
