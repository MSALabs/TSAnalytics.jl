# Handoff: Stage 10.1 — VAR and lag-order selection

**Status: pending.**

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first** — it carries the stage-wide
conventions (§5), the architecture decisions (§6), the reference
environment (§3) and the performance and test policy (§7, §8). This
document does not repeat them; it cites them.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document**, so re-establish a Julia runtime
and re-verify any package-internal claim before building on it.

---

## 1. Where this fits

- **Depends on:** Stage 1.4 (`_ols`, `src/unitroot.jl:51`) and
  **10.0** — multi-response OLS, `tsmatrix`/`tsnames`, companion matrix.
  10.0 is a new roadmap row; see the charter §6.2 and §2.1 of
  `stage-10-dependency-list.md`. **Do not start 10.1 before 10.0
  exists** or a loop of `_ols` calls gets baked in.
- **Feeds into:** 10.2 (Granger), 10.3 (IRF/FEVD), 10.7 (SVAR), 10.8
  (multivariate diagnostics). Every one of them consumes a fitted
  `VARModel`, so this item's field layout is load-bearing for four
  others.
- **Does not feed:** 10.4a, 10.4b, 10.5. Those need only `_ols` and
  10.0 — the roadmap's "10.5 depends on 10.1" is wrong.

---

## 2. Scope

### In

- `fit_var(Y, p; deterministic=:const)` → `VARModel`
- Five deterministic specifications spanning both references' vocabularies
- Coefficient standard errors, `t` statistics, two-sided p-values
- Both Σ conventions (df-adjusted and ML)
- Gaussian log-likelihood; `aic`/`bic`/`hqic`/`fpe` on the fitted model
- `var_select(Y; lag_max, deterministic)` → selection table across lags
- `roots` / `is_stable`
- `forecast` / `predict` (exact aliases), point + intervals
- Full StatsAPI contract: `coef`, `vcov`, `residuals`, `fitted`,
  `predict`, `loglikelihood`, `aic`, `bic`, `nobs`, `dof`
- Honest refusals for every nonsense input both references wave through

### Out (deliberately, stated rather than hidden)

- **Exogenous regressors (VARX).** Both references support it
  (`vars::VAR(exogen=)`, statsmodels `VAR(exog=)`). Not in the roadmap
  row. It is a clean later addition that reuses this item's `Z`
  construction untouched — flag it in the docstring as not implemented,
  do not half-build it.
- **Restricted VAR** (`vars::restrict`) — zero restrictions on
  individual coefficients.
- **Bayesian / shrinkage priors.**
- `k = 1`. Refused by name, matching both references. Charter §5.9.
- IRF, FEVD, Granger causality, residual diagnostics — 10.2, 10.3, 10.8.

---

## 3. Fixtures

All seeded and deterministic. `scripts/gen_fixtures.R` in this bundle
regenerates them bit-for-bit; `fixtures/` holds the CSVs and the
reference JSON. Copy the CSVs to `test/fixtures/var/` and the scripts to
`test/verification/var/`.

| File | Shape | DGP / provenance |
|---|---|---|
| `var2_p1.csv` | 200 × 2 | VAR(1), `A₁ = [0.5 −0.1; 0.2 0.3]`, `ν = [0.4, −0.2]`, `Σ = [1.0 0.3; 0.3 0.75]`, seed 20261013, 100 burn-in. `[V]` `A₁` read off the `byrow=TRUE` construction in `gen_fixtures.R` and confirmed by recovery — see §7.16 |
| `var3_p2.csv` | 250 × 3 | VAR(2), stationary, seed 20261014, 150 burn-in — **true order 2**, so selection has something to find |
| `var1_k1.csv` | 200 × 1 | AR(1), `φ = 0.6`, intercept 0.3, seed 20261015 — the `k = 1` refusal and the 10.0 reduction test |
| `var2_short.csv` | 30 × 2 | first 30 rows of `var2_p1` — df edge cases |
| `var2_nearunit.csv` | 200 × 2 | two AR(1)s at `φ = 0.98`, seed 20261016 — near-boundary stability |
| `var2_explosive.csv` | 60 × 2 | `A = [1.05 0.02; 0.00 1.03]`, seed 20261017 — `is_stable == false` |
| `uschange.csv` | 187 × 5 | fpp3 benchmark, already in `test_data/`, already exposed as `TSAnalytics.US_CHANGE` |
| `econ5.csv` | 160 × 5 | Shumway & Stoffer, already in `data/`, GPL-3, values of order 10⁴ — the conditioning stress case |

**Why `econ5` matters.** `[V]` It is the only fixture where
reference-vs-reference agreement degrades from `~1e-16` to `~5e-13` on
coefficients and `~2.7e-7` on FPE (whose values are of order `4.9e7`, so
relative agreement is still `~5e-15`). It is the fixture that will catch
a Cholesky-vs-QR conditioning mistake. Keep it, and set tolerances per
fixture rather than globally.

---

## 4. Verified references

`[V]` `vars` **1.6.1** on **R 4.3.3**; `statsmodels` **0.15.0**.
Environment setup: charter §3.

```r
library(vars)
v <- VAR(Y, p = 1, type = "const")
Bcoef(v)                      # K x (Kp + ndet), deterministic LAST
crossprod(resid(v)) / (v$obs - ncol(v$datamat) + ncol(Y))   # df-adjusted Sigma
crossprod(resid(v)) / v$obs                                  # ML Sigma
as.numeric(logLik(v))
lapply(v$varresult, function(e) coef(summary(e)))            # per-equation est/se/t
roots(v)                      # companion eigenvalue MODULI -- stable if < 1
predict(v, n.ahead = 5, ci = 0.95)
VARselect(Y, lag.max = 8, type = "const")
```

```python
from statsmodels.tsa.api import VAR
r = VAR(Y).fit(1, trend="c")
r.params        # (ndet + Kp) x K, deterministic FIRST, transposed vs R
r.coefs         # (p, K, K)
r.intercept; r.stderr; r.tvalues
r.sigma_u       # df-adjusted;  r.sigma_u_mle  # ML
r.llf; r.aic; r.bic; r.hqic; r.fpe
r.roots         # characteristic-polynomial roots -- stable if |.| > 1
r.forecast(Y[-p:], 5); r.forecast_interval(Y[-p:], 5, alpha=0.05); r.forecast_cov(5)
VAR(Y).select_order(8, trend="c")
```

### Conventions settled in the charter, applied here

| Topic | Charter | Decision for 10.1 |
|---|---|---|
| Coefficient layout | §5.1 | `A::Vector{Matrix{Float64}}` length `p`, each `K×K`; `deterministic::Matrix{Float64}` `K×ndet`. Neither reference's flat layout |
| Trend vocabulary | §5.2 | `deterministic::Symbol ∈ (:none, :const, :trend, :both, :quadratic)` |
| `roots` | §5.3 | companion moduli, stable if `< 1` (R's convention) |
| IC sample | §5.4 | selection holds `T = n − lag_max` fixed; fitted model uses its own `T`. Both pinned |
| Σ | §5.5 | store both |
| Nonsense inputs | §5.8 | refuse by name |
| `k = 1` | §5.9 | refuse by name |

---

## 5. Proposed Julia API

```julia
struct VARModel <: TimeSeriesModel
    # data
    Y::Matrix{Float64}                  # n × K, observations × variables
    names::Vector{Symbol}               # length K
    # fit
    p::Int
    deterministic::Symbol               # :none | :const | :trend | :both | :quadratic
    A::Vector{Matrix{Float64}}          # p matrices, each K × K; A[j][i,m] = eq i on y_m lag j
    det_coef::Matrix{Float64}           # K × ndet, columns in order (const, trend, trend^2)
    # second moments
    sigma_df::Matrix{Float64}           # E'E / (T - Kp - ndet)  -> se, intervals
    sigma_mle::Matrix{Float64}          # E'E / T                 -> loglik, IC
    se::Matrix{Float64}                 # K × (Kp + ndet), same layout as the stacked B
    vcov_full::Matrix{Float64}          # K(Kp+ndet) × K(Kp+ndet), kron(sigma_df, (Z'Z)^-1)
    # fit products
    resid::Matrix{Float64}              # T × K
    fitted::Matrix{Float64}             # T × K
    loglik::Float64
    T::Int                              # effective observations = n - p
    converged::Bool                     # always true; OLS, kept for contract uniformity
end

fit_var(Y, p::Integer; deterministic::Symbol = :const, method::Symbol = :qr) -> VARModel

var_select(Y; lag_max::Integer = 10, deterministic::Symbol = :const,
           parallel::Bool = true) -> VARSelection

struct VARSelection
    lags::Vector{Int}                   # 1:lag_max
    aic::Vector{Float64}
    bic::Vector{Float64}
    hqic::Vector{Float64}
    fpe::Vector{Float64}
    selected::NamedTuple                # (; aic, bic, hqic, fpe)
    T::Int                              # n - lag_max, held FIXED across lags
    deterministic::Symbol
end

roots(m::VARModel)      -> Vector{Float64}   # companion eigenvalue moduli
is_stable(m::VARModel)  -> Bool              # maximum(roots(m)) < 1
companion(m::VARModel)  -> Matrix{Float64}   # Kp × Kp, exported: 10.3 and 10.7 need it

forecast(m::VARModel, h::Integer; levels = [80.0, 95.0]) -> VARForecast
predict(m::VARModel, h::Integer; kwargs...)                        # EXACT alias
```

`vcov_full` is `kron(sigma_df, (Z'Z)⁻¹)` — needed in full, not as a
diagonal, because **10.3's analytic IRF standard errors consume it
whole** as `cov_a`. Stage 9B Tier 2.2 already established that storing a
full covariance is a fit-path change rather than an accessor; `ARXModel`
is `[S]` the only existing type that keeps one. `VARModel` is the
second, and it must be, so do not store only `se`.

### StatsAPI contract

All of `coef`, `vcov`, `residuals`, `fitted`, `predict`,
`loglikelihood`, `aic`, `bic`, `nobs`, `dof` — the fullest contract of
any model in the package. `[S]` For reference, the current coverage
gaps: `ArmaModel`, `ArimaxModel`, `SarimaxModel` and
`ExponentialSmoothingModel` implement almost none of it. Do not add a
tenth partially-conforming type.

`coef(m)` returns the stacked `K × (Kp + ndet)` matrix in **this
package's** order — lag blocks first, deterministic last (R's order,
since it reads naturally) — with `coefnames(m)` giving
`[:y1_l1, :y2_l1, …, :const, :trend]`.

---

## 6. Implementation

### 6.1 Regressor construction

For `p` lags and `ndet` deterministic columns, with `T = n − p`:

```
Z = [ y_{t-1}' … y_{t-p}' | d_t' ]   for t = p+1 … n       (T × (Kp + ndet))
X = [ y_t' ]                          for t = p+1 … n       (T × K)
```

Build `Z` **once**, column-major, with `@views` into `Y`. The
deterministic columns are appended last.

`[V]` **The trend column runs `p+1 : n`, not `1 : T`** — the calendar
index of the estimation sample, not a within-sample counter. Settled by
fitting both forms directly against the reference coefficients on
`var2_p1.csv`, `p = 1`, `:both`:

```
trend = p+1:n    |const - ref| = 1.97e-15      <- matches
trend = 1:T      |const - ref| = 4.922e-04
```

The trend *coefficient* is identical either way (`4.03e-17`); the origin
shift is absorbed entirely by the constant, and the discrepancy is
exactly `p ×` the trend coefficient. So the fit, residuals and
log-likelihood are the same under both forms — only the reported
intercept differs, which is why this is easy to get wrong silently.

`[V]` R's `type="both"` and statsmodels' `trend="ct"` agree with each
other on both deterministic coefficients to `9.5e-16`, so there is a
single right answer here rather than a convention to choose.

### 6.2 One factorization, K solves

This is the whole performance argument for the item:

```julia
F = qr(Z)                      # ONE factorization, O(T(Kp+ndet)^2)
B = F \ X                      # K solves share it -> (Kp+ndet) × K
E = X - Z * B                  # T × K residuals, one GEMM
```

`Z \ X` with a matrix right-hand side already does exactly this in
Julia — one QR, `K` back-substitutions — so the naive-looking expression
is the fast one. **Do not loop `_ols` over columns**: that re-factorizes
`Z` `K` times for no benefit. Extend `_ols` to accept an
`AbstractMatrix` response in 10.0 and call that, so the GLS/Cholesky
options and the error messages stay in one place.

`(Z'Z)⁻¹` for the standard errors comes from the same `F`:
`Rinv = F.R \ I(k)`, `ZtZ_inv = Rinv * Rinv'` — the pattern `_ols`
already uses (`[S]` `src/unitroot.jl:73–78`). Never form `Z'Z` and
invert it when `method=:qr`.

### 6.3 Log-likelihood

```julia
loglik = -T*K/2 * log(2π) - T/2 * logdet(sigma_mle) - T*K/2
```

`[V]` This matches both references exactly — `−500.8313597100677` (R)
vs `−500.8313597100678` (statsmodels) on `var2_p1`, p=1, const.

Use `logdet(cholesky(Hermitian(sigma_mle)))`, not `log(det(...))`:
`det` overflows for `K` large with values of `econ5`'s magnitude, and the
Cholesky also asserts positive-definiteness, which is the thing that
fails first when `p` is too large.

### 6.4 Information criteria

`[V]` Verified to `1e-12` against both references at p = 1, 2, 3, 8 with
`nfree = Kp + ndet` **per equation** and `Σ̃` the **ML** divisor:

```julia
ld = logdet(sigma_mle); npar = K * (K*p + ndet)
aic  = ld + 2/T * npar
bic  = ld + log(T)/T * npar
hqic = ld + 2*log(log(T))/T * npar
fpe  = ((T + K*p + ndet) / (T - K*p - ndet))^K * det(sigma_mle)
```

`fpe` is the one place `det` rather than `logdet` is required, because
the reference reports the determinant itself. On `econ5` `[V]` its value
is `4.9e7` — fine in Float64, but note it in the docstring.

### 6.5 Order selection

Hold the sample fixed (charter §5.4): for every candidate `j`, fit on
`Y[lag_max - j + 1 : end]` so that every fit has `T = n − lag_max`.

The candidates are independent, so `Threads.@threads` over `j` — the
pattern `auto_ets` already uses. Each thread needs its own `Z`
buffer; do not share one.

### 6.6 Companion matrix, roots, stability

```
C = [ A₁ A₂ … A_p ]        Kp × Kp
    [ I   0  …  0  ]
    [ 0   I  …  0  ]
    [ …            ]
roots(m)    = sort!(abs.(eigvals(C)); rev = true)
is_stable(m) = maximum(roots(m)) < 1
```

Build `C` once and cache it on first use — 10.3's IRF recursion and
10.7 both want it. `eigvals` on a general real matrix returns
`Complex{Float64}`; take `abs` and return `Vector{Float64}`.

### 6.7 Forecast

Deterministic recursion, then the MSE accumulation:

```
ŷ_{n+h} = d_{n+h}'Θ + Σ_{j=1}^{p} A_j ŷ_{n+h-j}        (ŷ = y for h ≤ 0)
Σ_h     = Σ_{i=0}^{h-1} Φ_i Σ Φ_i'                      Φ_0 = I, Φ_i from the MA rep
```

`[V]` Both references use **Σ_df** here and **ignore parameter
estimation uncertainty** — confirmed because `forecast_cov(1)` equals
`sigma_u` element-for-element (`1.024427598172` for `[1,1]` on
`var2_p1`), and R's `predict` intervals match statsmodels'
`forecast_interval` to `1.1e-15` (lower) and `1.3e-15` (upper). Follow
them, and say in the docstring that the intervals are conditional on the
estimated parameters — the same honesty as the `fit_ets` `sigma2`
denominator note.

The `Φ_i` recursion is shared with 10.3's `ma_rep`. Write it once, in
10.1, and have 10.3 call it.

### 6.8 Performance requirements

Per charter §7. Concretely for this item:

1. `@code_warntype` clean on `fit_var`, `var_select`, `forecast`.
2. `Vector{Matrix{Float64}}` for `A`, never `Vector{Matrix}`.
3. The forecast recursion allocates **zero** bytes after setup —
   preallocate `ŷ`, use `mul!` and `@views`, assert with `@allocated`.
4. `Z` built with `@inbounds` and `@views`; one allocation for the
   matrix itself.
5. `var_select` parallel and serial must select the identical model —
   asserted, not assumed.
6. Benchmark table in the handoff when done, for `k ∈ {2, 5, 10}` and
   `p ∈ {1, 2, 4, 8}`, against `vars::VAR` and statsmodels. One `qr`
   plus `K` solves should beat `vars`, which `[S]` fits `K` separate
   `lm()` calls (`vars/R/VAR.R`, 94 lines, delegating to `lm`).

---

## 7. Test battery

Exact values below are all `[V]`. The full reference set is in
`fixtures/ref_R.json`, `ref_py.json`, `ref_R_extra.json`,
`ref_py_extra.json`; the generating scripts are in `scripts/`. Load the
JSON in the test file rather than transcribing everything by hand — but
keep the headline numbers inline as below, so a reader can see what is
being asserted without opening a data file.

Suggested file: `test/test_var.jl`. Target: **≥ 300 assertions**.

### 7.1 Core fit — `var2_p1.csv`, k=2, p=1, `:const`

```julia
@testset "fit_var: var2_p1, p=1, const -- vs R and Python" begin
    Y = readdlm(joinpath(FIXTURES, "var2_p1.csv"), ',', Float64; skipstart=1)
    m = fit_var(Y, 1; deterministic=:const)

    # A1: rows = equations, cols = variables           (R Bcoef[:, 1:2])
    @test m.A[1] ≈ [0.473845644615  0.013318881594
                    0.237738422675  0.385842223937]  atol=1e-11
    # intercept                                        (R Bcoef[:, 3])
    @test m.det_coef[:, 1] ≈ [0.417793672593, -0.229887626602] atol=1e-11

    @test m.sigma_df ≈ [1.024427598172  0.307967575221
                        0.307967575221  0.62197408186 ] atol=1e-11
    @test m.sigma_mle ≈ [1.008983966038  0.303324847956
                         0.303324847956  0.612597588163] atol=1e-11

    @test m.loglik ≈ -500.8313597100677 atol=1e-9      # R: ...677, py: ...678
    @test m.T == 199
    @test nobs(m) == 199

    # standard errors, R's per-equation order [y1.l1, y2.l1, const]
    @test m.se[1, :] ≈ [0.072925266989, 0.0869002758,   0.094054800935] atol=1e-11
    @test m.se[2, :] ≈ [0.056822918606, 0.067712159345, 0.073286921237] atol=1e-11

    @test residuals(m)[1:3, :] ≈ [ 1.978220647824  1.296722423622
                                  -1.736741784841 -0.595056772665
                                  -1.721864370948 -0.019034264545] atol=1e-10

    # fitted + resid must reconstruct the data exactly
    @test fitted(m) .+ residuals(m) ≈ Y[2:end, :] atol=1e-12
end
```

### 7.2 Fitted-model information criteria

```julia
@testset "fitted-model IC (own sample, T=199)" begin
    m = fit_var(Y, 1; deterministic=:const)
    @test aic(m)  ≈ -0.5819716231697688 atol=1e-12
    @test bic(m)  ≈ -0.482676000313754  atol=1e-12
    @test hqic(m) ≈ -0.5417841187650027 atol=1e-12
    @test fpe(m)  ≈ 0.5587980985371741  atol=1e-12
    # NOT equal to var_select's value at lag 1 -- different sample. See charter 5.4.
    @test !isapprox(aic(m), -0.6050354269424555; atol=1e-6)
end
```

### 7.3 Order selection — the sample-fixing convention

```julia
@testset "var_select: var2_p1, lag_max=8, const" begin
    s = var_select(Y; lag_max=8, deterministic=:const)
    @test s.T == 192                       # n - lag_max, NOT n - p
    @test s.lags == 1:8
    @test s.aic ≈ [-0.605035426942, -0.573198953406, -0.555219028246, -0.53071870995,
                   -0.491890767193, -0.457493178216, -0.421144104509, -0.409585305809] atol=1e-11
    @test s.hqic ≈ [-0.563807005172, -0.504484917121, -0.459019377448, -0.407033444637,
                    -0.340719887367, -0.278836683875, -0.215001995655, -0.175957582441] atol=1e-11
    @test s.bic ≈ [-0.503238696567, -0.403537736113, -0.317693324036, -0.225328518822,
                   -0.118636089148, -0.016374013254,  0.087839547371,  0.167262832988] atol=1e-11
    @test s.fpe ≈ [0.546057859029, 0.563732519432, 0.573983652928, 0.58826297215,
                   0.611622792689, 0.633130849485, 0.656714449305, 0.664543183561] atol=1e-11
    @test s.selected == (aic=1, bic=1, hqic=1, fpe=1)   # R VARselect: [1,1,1,1]

    # a fit at the selection's own sample reproduces the selection's number
    m192 = fit_var(Y[8:end, :], 1; deterministic=:const)
    @test aic(m192) ≈ s.aic[1] atol=1e-12
end
```

### 7.4 Order selection recovers the true order — `var3_p2.csv`

```julia
@testset "var_select: var3_p2 (true order 2), k=3" begin
    Y3 = read_fixture("var3_p2.csv")                      # 250 × 3
    s = var_select(Y3; lag_max=8, deterministic=:const)
    @test s.selected == (aic=2, bic=1, hqic=2, fpe=2)     # R: [2, 2, 1, 2]
    @test s.aic[1:4] ≈ [-0.063748812963, -0.139883826322,
                        -0.102824171152, -0.061898645218] atol=1e-11
    @test s.bic[1:4] ≈ [0.109256363541, 0.16287523256,
                        0.329688770107, 0.500368178419]   atol=1e-11
    @test s.fpe[1:4] ≈ [0.938249119047, 0.869501344961,
                        0.902413076163, 0.940270132509]   atol=1e-11
    @test argmin(s.aic) == 2                               # AIC finds the DGP
end
```

**Note the disagreement worth documenting:** `[V]` BIC picks 1 where the
true order is 2, on 250 observations. That is BIC being BIC, not a bug —
assert it and say so in the manual.

### 7.5 Higher order — `var3_p2`, p=2 and p=3

```julia
@testset "fit_var: var3_p2, p=2, const" begin
    m = fit_var(Y3, 2; deterministic=:const)
    @test m.loglik ≈ -1017.2884090692235 atol=1e-8
    @test m.A[1] ≈ [ 0.290077426458   0.068287895814  -0.030404069817
                    -0.045247308605   0.557104755017   0.099593849045
                    -0.1257081212     0.205459393486   0.508486961624] atol=1e-10
    @test m.A[2] ≈ [ 0.191143749358  -0.019570758286   0.137891937175
                    -0.000107435400   0.093802396263   0.049493870786
                     0.244293832209  -0.19976354471    0.162354789018] atol=1e-10
    @test m.det_coef[:, 1] ≈ [0.031130809530, 0.124867937320, -0.065545311252] atol=1e-10
    @test m.sigma_df ≈ [ 1.112686577915  0.353447661619 -0.167777712865
                         0.353447661619  0.755620685823  0.05875940018
                        -0.167777712865  0.05875940018   1.161647845728] atol=1e-10
end

@testset "fit_var: var2_p1 at p=2 (over-fitted)" begin
    m = fit_var(Y, 2; deterministic=:const)
    @test m.loglik ≈ -496.03769349818344 atol=1e-8
    @test m.A[1] ≈ [0.448249746032 0.070329548445
                    0.220514094415 0.397525035175] atol=1e-11
    @test m.A[2] ≈ [0.014232369339 -0.111386497811
                    0.013417589992 -0.02905686488 ] atol=1e-11
    @test m.det_coef[:, 1] ≈ [0.414053755176, -0.234406837193] atol=1e-11
end
```

### 7.6 All five deterministic specifications

```julia
@testset "deterministic specifications" begin
    @test fit_var(Y, 1; deterministic=:none).loglik  ≈ -522.3719059321885 atol=1e-9
    @test fit_var(Y, 1; deterministic=:const).loglik ≈ -500.8313597100677 atol=1e-9
    @test fit_var(Y, 1; deterministic=:trend).loglik ≈ -506.6828693867048 atol=1e-9
    @test fit_var(Y, 1; deterministic=:both).loglik  ≈ -500.5285172458806 atol=1e-9
    @test fit_var(Y, 1; deterministic=:quadratic).loglik ≈ -497.61123010854993 atol=1e-9

    # :none -- no deterministic column at all
    mn = fit_var(Y, 1; deterministic=:none)
    @test size(mn.det_coef) == (2, 0)
    @test mn.A[1] ≈ [0.682877213721 -0.130186512989
                     0.122720473724  0.464804919471] atol=1e-11

    # :trend -- trend only, NO constant (R type="trend"; statsmodels has no equivalent)
    mt = fit_var(Y, 1; deterministic=:trend)
    @test size(mt.det_coef) == (2, 1)
    @test mt.A[1] ≈ [0.537421766282 -0.02137156876
                     0.206048280467  0.40246754536 ] atol=1e-11
    @test mt.det_coef[:, 1] ≈ [0.002847613152, -0.001631326723] atol=1e-11

    # :both == statsmodels trend="ct"; const then trend
    mb = fit_var(Y, 1; deterministic=:both)
    @test size(mb.det_coef) == (2, 2)
    @test mb.det_coef[:, 1] ≈ [ 0.368606055009, -0.185002094391] atol=1e-11
    @test mb.det_coef[:, 2] ≈ [ 0.000492205472, -0.000449155817] atol=1e-11

    # :quadratic == statsmodels trend="ctt"; const, trend, trend^2
    mq = fit_var(Y, 1; deterministic=:quadratic)
    @test size(mq.det_coef) == (2, 3)
    @test mq.det_coef[:, 1] ≈ [ 0.800541127082,  0.025360531334] atol=1e-10
    @test mq.det_coef[:, 2] ≈ [-0.011490295757, -0.006284918144] atol=1e-10
    @test mq.det_coef[:, 3] ≈ [ 5.913177572457e-5, 2.879857740094e-5] atol=1e-13
    @test mq.A[1] ≈ [0.446047669246 -0.018784843077
                     0.224944820795  0.367529114583] atol=1e-10
end
```

**The trap this testset exists for** (charter §5.2): `:trend` and
`:both` differ by `6.15` in log-likelihood. R's `type="trend"` and
statsmodels' `trend="ct"` sound equivalent and are not. Assert the gap
explicitly:

```julia
@test fit_var(Y,1;deterministic=:trend).loglik <
      fit_var(Y,1;deterministic=:both).loglik - 6.0
```

### 7.7 Order selection under each deterministic spec

```julia
@testset "var_select: deterministic variants" begin
    @test var_select(Y; lag_max=8, deterministic=:none).selected  == (aic=2, bic=1, hqic=1, fpe=2)
    @test var_select(Y; lag_max=8, deterministic=:const).selected == (aic=1, bic=1, hqic=1, fpe=1)
    @test var_select(Y; lag_max=8, deterministic=:trend).selected == (aic=1, bic=1, hqic=1, fpe=1)

    sn = var_select(Y; lag_max=8, deterministic=:none)
    @test sn.aic[1:3] ≈ [-0.406555, -0.428435, -0.412922] atol=1e-6
    st = var_select(Y; lag_max=8, deterministic=:trend)
    @test st.aic[1:3] ≈ [-0.542931, -0.520116, -0.496543] atol=1e-6
    # :trend (R) vs :both (= statsmodels "ct") differ -- the mismatch propagates
    @test abs(st.aic[1] - (-0.585634)) > 0.04     # statsmodels trend="ct" value
end
```

### 7.8 Real data — `uschange` (k=5) and `econ5` (k=5, large values)

```julia
@testset "fit_var: uschange, p=2, const" begin
    U = read_fixture("uschange.csv")                       # 187 × 5
    m = fit_var(U, 2; deterministic=:const)
    @test m.loglik ≈ -1174.1609283211412 atol=1e-7
    @test m.sigma_df[1, :] ≈ [0.369574475216, 0.199516463793, 0.379682807764,
                             -1.899479226109, -0.092305049392] atol=1e-9
    f = forecast(m, 8)
    @test f.point[1, :] ≈ [ 0.793285184354,  0.553438191632,  0.531335915345,
                           -0.541365385204, -0.037877336534] atol=1e-9
    @test f.point[2, :] ≈ [ 0.74962587236,   0.792494400384,  0.633115405576,
                            2.20100024683,  -0.02225935201 ] atol=1e-9
end

@testset "fit_var: econ5, p=2 -- conditioning stress (values ~1e4)" begin
    E = read_fixture("econ5.csv")                          # 160 × 5
    m = fit_var(E, 2; deterministic=:const)
    @test m.loglik ≈ -2451.9592961449634 atol=1e-6         # looser: see 3
    # QR and Cholesky must still agree here -- this is the fixture that catches it
    mc = fit_var(E, 2; deterministic=:const, method=:cholesky)
    @test mc.A[1] ≈ m.A[1] atol=1e-8
    @test mc.loglik ≈ m.loglik atol=1e-6
end

@testset "var_select: real data" begin
    @test var_select(read_fixture("uschange.csv"); lag_max=8).selected ==
          (aic=3, bic=1, hqic=1, fpe=3)                    # R VARselect: [3, 1, 1, 3]
    @test var_select(read_fixture("econ5.csv");    lag_max=6).selected ==
          (aic=5, bic=2, hqic=1, fpe=5)                    # R VARselect: [5, 2, 1, 5]
end
```

`[V]` Note `econ5`'s FPE values are of order `3e7`–`5e7`
(`[4.935167943089e7, 3.324183786901e7, 3.468846706719e7, …]`); assert
them with `rtol`, not `atol`.

### 7.9 `roots` and stability — including the reciprocal convention

```julia
@testset "roots: companion moduli, and the reciprocal of statsmodels" begin
    m = fit_var(Y, 1; deterministic=:const)
    @test roots(m) ≈ [0.501276142801, 0.358411725751] atol=1e-11   # sorted desc
    @test is_stable(m)
    # statsmodels returns characteristic roots: exact reciprocals
    sm_roots = [1.99490842, 2.79008729]
    @test sort(roots(m)) ≈ sort(1 ./ sm_roots) atol=1e-7

    m2 = fit_var(Y, 2; deterministic=:const)
    @test sort(roots(m2)) ≈ [0.0659202, 0.0659202, 0.49876087, 0.49876087] atol=1e-7

    nu = fit_var(read_fixture("var2_nearunit.csv"), 1)
    @test roots(nu) ≈ [0.974494137410, 0.871893192744] atol=1e-10
    @test is_stable(nu)                       # 0.974 < 1: stable, but near the boundary
    @test nu.loglik ≈ -538.4680911382525 atol=1e-9

    ex = fit_var(read_fixture("var2_explosive.csv"), 1)
    @test roots(ex) ≈ [1.040418520983, 0.941776653648] atol=1e-10
    @test !is_stable(ex)
    @test ex.loglik ≈ -156.2799235045827 atol=1e-9
end

@testset "companion matrix structure" begin
    m2 = fit_var(Y, 2; deterministic=:const)
    C = companion(m2)
    @test size(C) == (4, 4)
    @test C[1:2, 1:2] ≈ m2.A[1] atol=1e-14
    @test C[1:2, 3:4] ≈ m2.A[2] atol=1e-14
    @test C[3:4, 1:2] ≈ I(2)
    @test all(iszero, C[3:4, 3:4])
    @test sort(abs.(eigvals(C))) ≈ sort(roots(m2)) atol=1e-12
end
```

### 7.10 Forecast point, intervals, and the Σ convention

```julia
@testset "forecast: var2_p1, p=1, h=5" begin
    m = fit_var(Y, 1; deterministic=:const)
    f = forecast(m, 5; levels=[95.0])

    @test f.point ≈ [1.563745617857  0.097324529068
                     1.16006397678   0.179426702792
                     0.969874698474  0.115134551479
                     0.878898037746  0.045112625801
                     0.834856529561 -0.003533437549] atol=1e-10

    @test f.lower[:, :, 1] ≈ [-0.420012541082 -1.448407602642
                              -1.038627405873 -1.605076774773
                              -1.276178114978 -1.749695720626
                              -1.378309714499 -1.845790168218
                              -1.425042547875 -1.902333186945] atol=1e-10
    @test f.upper[:, :, 1] ≈ [3.547503776797 1.643056660779
                              3.358755359432 1.963930180358
                              3.215927511927 1.979964823584
                              3.136105789991 1.936015419819
                              3.094755606997 1.895266311847] atol=1e-10

    # h=1 forecast covariance IS sigma_df -- parameter uncertainty is NOT included,
    # matching both references (verified: statsmodels forecast_cov(1) == sigma_u)
    @test f.cov[1] ≈ m.sigma_df atol=1e-12

    # intervals are symmetric about the point forecast
    @test f.upper[:, :, 1] .- f.point ≈ f.point .- f.lower[:, :, 1] atol=1e-10

    # h=1 point forecast equals the one-step recursion by hand
    @test f.point[1, :] ≈ m.det_coef[:, 1] .+ m.A[1] * Y[end, :] atol=1e-12
end

@testset "forecast: var3_p2, p=2" begin
    m = fit_var(Y3, 2; deterministic=:const)
    f = forecast(m, 5)
    @test f.point[1, :] ≈ [0.767262021951, 1.333370549241, 1.378146754024] atol=1e-10
    @test f.point[2, :] ≈ [1.019556596221, 1.252721686486, 1.417641280056] atol=1e-10
end

@testset "predict is an exact alias for forecast" begin
    m = fit_var(Y, 1)
    a = forecast(m, 5); b = predict(m, 5)
    @test a.point == b.point        # identical, not just approx
    @test a.lower == b.lower
    @test a.upper == b.upper
    @test typeof(a) === typeof(b)
end
```

### 7.11 Both Σ conventions, both references

```julia
@testset "sigma conventions" begin
    for (fx, p) in [("var2_p1.csv",1), ("var3_p2.csv",2),
                    ("uschange.csv",2), ("econ5.csv",2), ("var2_short.csv",2)]
        Yf = read_fixture(fx); m = fit_var(Yf, p; deterministic=:const)
        K = size(Yf, 2); T = m.T; ndet = 1
        # the two differ by exactly the df factor
        @test m.sigma_df ≈ m.sigma_mle .* (T / (T - K*p - ndet)) rtol=1e-12
        @test issymmetric(round.(m.sigma_df, digits=12))
        @test isposdef(Hermitian(m.sigma_df))
        # sigma_mle is what the log-likelihood uses
        @test m.loglik ≈ -T*K/2*log(2π) - T/2*logdet(m.sigma_mle) - T*K/2 rtol=1e-12
    end
end
```

### 7.12 Minimum viable sample — `var2_short.csv` (n=30)

```julia
@testset "short sample: n=30, k=2, p=2" begin
    S = read_fixture("var2_short.csv")
    m = fit_var(S, 2; deterministic=:const)
    @test m.T == 28
    @test m.loglik ≈ -63.12030284334104 atol=1e-10
    @test m.A[1] ≈ [0.62186346997  -0.307650746511
                    0.219862710709  0.177378739374] atol=1e-11
    @test m.A[2] ≈ [-0.296431841767 -0.181090171558
                    -0.056554840712  0.027660067304] atol=1e-11
    @test m.det_coef[:, 1] ≈ [0.890690612697, 0.006770459902] atol=1e-11
    @test sort(roots(m)) ≈ [0.16421091, 0.26842585, 0.64681086, 0.64681086] atol=1e-7
end
```

### 7.13 Honest refusals — where this package beats both references

`[V]` Both references silently accept the first two.

```julia
@testset "refusals" begin
    # p too large for the sample: R and statsmodels BOTH return "no error"
    @test_throws ArgumentError fit_var(Y[1:8, :], 5; deterministic=:const)
    e = try fit_var(Y[1:8, :], 5) catch err; err end
    @test occursin("observations", lowercase(sprint(showerror, e)))

    # p = 0: R errors with an internal dimnames message; statsmodels accepts it
    @test_throws ArgumentError fit_var(Y, 0)
    @test_throws ArgumentError fit_var(Y, -1)

    # NaN / Inf: R says "NAs in y."; statsmodels leaks LinAlgError from LAPACK
    Yn = copy(Y); Yn[5, 1] = NaN
    @test_throws ArgumentError fit_var(Yn, 1)
    Yi = copy(Y); Yi[7, 2] = Inf
    @test_throws ArgumentError fit_var(Yi, 1)

    # k = 1: both references refuse; so do we, and the message must point somewhere
    @test_throws ArgumentError fit_var(read_fixture("var1_k1.csv"), 1)
    e1 = try fit_var(read_fixture("var1_k1.csv"), 1) catch err; err end
    msg = sprint(showerror, e1)
    @test occursin("fit_arima", msg) || occursin("fit_arx", msg)

    # invalid keywords
    @test_throws ArgumentError fit_var(Y, 1; deterministic=:bogus)
    @test_throws ArgumentError fit_var(Y, 1; method=:bogus)
    @test_throws ArgumentError var_select(Y; lag_max=0)
    @test_throws ArgumentError var_select(Y; lag_max=200)      # lag_max >= n
end
```

### 7.14 Reduction: `k = 1` through the 10.0 primitive onto `ar.ols`

`fit_var` refuses `k = 1`, but 10.0's multi-response OLS must handle it,
and `stats::ar.ols` is exactly a `k = 1` VAR(p) with a constant.

```julia
@testset "10.0 reduction: k=1 multi-response OLS == ar.ols" begin
    z = read_fixture("var1_k1.csv")                 # 200 × 1
    B, E, se = _mols(build_var_Z(z, 1, :const), z[2:end, :])
    @test B[1, 1] ≈ 0.6484308401025832  atol=1e-11   # ar.ols ar
    @test B[2, 1] ≈ 0.32794476411707674 atol=1e-11   # ar.ols x.intercept
    @test se[1, 1] ≈ 0.05410479122534204 atol=1e-9   # ar.ols asy.se on ar
    @test se[2, 1] ≈ 0.08644379310823685 atol=1e-9   # ar.ols asy.se on intercept
    # var.pred is ar.ols's own residual variance
    @test dot(E, E) / (length(E) - 2) ≈ 0.9538204266778918 rtol=1e-9

    B2, = _mols(build_var_Z(z, 2, :const), z[3:end, :])
    @test B2[1:2, 1] ≈ [0.673140896685106, -0.03945717137380811] atol=1e-11
    @test B2[3, 1]   ≈ 0.33998905461259055 atol=1e-11
end
```

### 7.15 Reduction: a VAR with a diagonal `A` is `K` independent ARs

```julia
@testset "reduction: diagonal VAR == K univariate AR fits" begin
    # var2_nearunit is two INDEPENDENT AR(1)s -- off-diagonals should be ~0
    N = read_fixture("var2_nearunit.csv")
    m = fit_var(N, 1; deterministic=:const)
    @test abs(m.A[1][1, 2]) < 0.1
    @test abs(m.A[1][2, 1]) < 0.1
    # each diagonal element must match an independent univariate OLS AR(1)
    for j in 1:2
        b, = TSAnalytics._ols(hcat(N[1:end-1, j], ones(size(N,1)-1)), N[2:end, j])
        @test m.A[1][j, j] ≈ b[1] atol=0.05     # loose: the VAR conditions on both
    end
end
```

### 7.16 Parameter recovery from the known DGP

`[V]` The true generating matrix is `A₁ = [0.5 −0.1; 0.2 0.3]` and
`ν = [0.4, −0.2]`, read off `gen_fixtures.R`'s `byrow=TRUE` construction
from `c(0.5, -0.1, 0.2, 0.3)`. Recovery, in standard-error units:

| Parameter | True | Estimated | se | Gap (se) |
|---|---|---|---|---|
| `A₁[1,1]` | 0.5 | 0.473845644615 | 0.072925266989 | 0.36 |
| `A₁[1,2]` | −0.1 | 0.013318881594 | 0.086900275800 | 1.30 |
| `A₁[2,1]` | 0.2 | 0.237738422675 | 0.056822918606 | 0.66 |
| `A₁[2,2]` | 0.3 | 0.385842223937 | 0.067712159345 | 1.27 |
| `ν[1]` | 0.4 | 0.417793672593 | 0.094054800935 | 0.19 |
| `ν[2]` | −0.2 | −0.229887626602 | 0.073286921237 | 0.41 |

All six inside 1.31 se — normal finite-sample MLE error at n = 200,
nothing to explain away.

```julia
@testset "parameter recovery: var2_p1, true A1 = [0.5 -0.1; 0.2 0.3]" begin
    m = fit_var(Y, 1; deterministic=:const)
    A_true = [0.5 -0.1; 0.2 0.3];  nu_true = [0.4, -0.2]
    se_A   = [0.072925266989 0.0869002758
              0.056822918606 0.067712159345]
    se_nu  = [0.094054800935, 0.073286921237]

    # every coefficient within 2 standard errors of truth
    @test all(abs.(m.A[1] .- A_true) .< 2 .* se_A)
    @test all(abs.(m.det_coef[:, 1] .- nu_true) .< 2 .* se_nu)
    # and the worst one is where it was measured: A1[1,2], 1.30 se
    @test maximum(abs.(m.A[1] .- A_true) ./ se_A) ≈ 1.304010609293 atol=1e-6

    # Sigma recovery: true [1.0 0.3; 0.3 0.75]
    @test m.sigma_df[1,1] ≈ 1.00 atol=0.15
    @test m.sigma_df[2,2] ≈ 0.75 atol=0.20
    @test m.sigma_df[1,2] ≈ 0.30 atol=0.15
end
```

**Also assert the trend origin**, since §6.1 shows the wrong choice is
silent in everything but the intercept:

```julia
@testset "trend column origin is p+1:n, not 1:T" begin
    m = fit_var(Y, 1; deterministic=:both)
    @test m.det_coef[:, 1] ≈ [ 0.368606055009, -0.185002094391] atol=1e-11
    @test m.det_coef[:, 2] ≈ [ 0.000492205472, -0.000449155817] atol=1e-13
    # the 1:T mistake shifts the constant by exactly p * trend_coef
    wrong = m.det_coef[:, 1] .+ m.p .* m.det_coef[:, 2]
    @test !isapprox(wrong, m.det_coef[:, 1]; atol=1e-6)
end
```

### 7.17 StatsAPI contract

```julia
@testset "StatsAPI contract" begin
    m = fit_var(Y, 1; deterministic=:const)
    @test size(coef(m)) == (2, 3)
    @test coefnames(m) == [:y1_l1, :y2_l1, :const]
    @test size(vcov(m)) == (6, 6)                       # K(Kp+ndet) square
    @test issymmetric(round.(vcov(m), digits=12))
    @test isposdef(Hermitian(vcov(m)))
    @test sqrt.(diag(vcov(m))) ≈ vec(m.se') atol=1e-11  # diag == stored se
    @test size(residuals(m)) == (199, 2)
    @test size(fitted(m))    == (199, 2)
    @test nobs(m) == 199
    @test dof(m)  == 6
    @test loglikelihood(m) ≈ m.loglik
    @test aic(m) ≈ -2*m.loglik/ m.T + 0 skip=true        # see 7.2: VAR IC are log-det based
    @test_nowarn show(IOBuffer(), m)                     # CoefTable print path
end
```

`vcov` must be `kron(sigma_df, (Z'Z)⁻¹)` in full — 10.3 consumes it
whole. Assert the Kronecker structure directly:

```julia
    ZtZinv = inv(Z' * Z)
    @test vcov(m) ≈ kron(m.sigma_df, ZtZinv) atol=1e-10
```

### 7.18 Performance and type stability

```julia
@testset "type stability and allocations" begin
    m = fit_var(Y, 1; deterministic=:const)
    @test eltype(m.A) === Matrix{Float64}               # NOT Matrix{<:Real}
    @test isconcretetype(typeof(m))
    @test (@inferred roots(m)) isa Vector{Float64}
    @test (@inferred is_stable(m)) isa Bool

    # the forecast recursion must not allocate after setup
    f = forecast(m, 10)                                  # warm up
    alloc = @allocated forecast(m, 10)
    @test alloc < 20_000                                 # tighten once measured

    # one factorization: fitting k equations must not cost k factorizations
    t2 = @elapsed fit_var(Y, 1)
    U  = read_fixture("uschange.csv")
    t5 = @elapsed fit_var(U, 1)
    @test t5 < 6 * t2                                    # sublinear in K, not K^2
end

@testset "var_select: parallel == serial" begin
    sp = var_select(Y3; lag_max=8, parallel=true)
    ss = var_select(Y3; lag_max=8, parallel=false)
    @test sp.aic == ss.aic                               # bit-identical
    @test sp.selected == ss.selected
end
```

### 7.19 Container-agnostic input (`tsmatrix`)

```julia
@testset "tsmatrix: container agnostic" begin
    ref = fit_var(Y, 1).A[1]
    @test fit_var((Y[:,1], Y[:,2]), 1).A[1] ≈ ref atol=1e-14        # tuple of vectors
    @test fit_var([Y[:,1], Y[:,2]], 1).A[1] ≈ ref atol=1e-14        # vector of vectors
    nt = (y1 = Y[:,1], y2 = Y[:,2])
    m = fit_var(nt, 1)
    @test m.A[1] ≈ ref atol=1e-14
    @test m.names == [:y1, :y2]                                      # names from NamedTuple
    @test fit_var(Y, 1).names == [:y1, :y2]                          # default names
    # ragged input must be refused
    @test_throws ArgumentError fit_var((Y[:,1], Y[1:end-1, 2]), 1)
end
```

### 7.20 QR vs Cholesky agreement

```julia
@testset "method=:qr vs :cholesky" begin
    for (fx, p) in [("var2_p1.csv",1), ("var3_p2.csv",2),
                    ("uschange.csv",2), ("econ5.csv",2)]
        Yf = read_fixture(fx)
        a = fit_var(Yf, p; method=:qr)
        b = fit_var(Yf, p; method=:cholesky)
        @test a.A[1] ≈ b.A[1] rtol=1e-8
        @test a.loglik ≈ b.loglik rtol=1e-10
        @test a.se ≈ b.se rtol=1e-7
    end
    # collinear input: :cholesky must fail informatively, :qr must not
    Yc = hcat(Y, Y[:, 1])                        # duplicated column
    @test_throws ArgumentError fit_var(Yc, 1; method=:cholesky)
end
```

### 7.21 Regression guards against the known traps

```julia
@testset "convention guards (charter 5.1-5.4)" begin
    m = fit_var(Y, 1; deterministic=:const)
    # 5.1: deterministic LAST in coef(), not first
    @test coef(m)[:, end] ≈ m.det_coef[:, 1] atol=1e-14
    # 5.3: roots are companion moduli (< 1 for stable), NOT characteristic roots
    @test all(roots(m) .< 1.0)
    @test maximum(roots(m)) < 0.51
    # 5.4: selection IC != fitted IC
    s = var_select(Y; lag_max=8)
    @test !isapprox(s.aic[1], aic(m); atol=1e-6)
    @test s.T == 192 && m.T == 199
    # 5.4: our lags vector starts at 1, matching VARselect, not select_order's 0
    @test first(s.lags) == 1 && length(s.lags) == 8
end
```

---

## 8. Checklist for the implementing session

1. **Re-establish Julia** and re-verify every `[S]` claim in §1, §5 and
   the charter §6 before writing code.
2. Build **10.0** first: multi-response `_ols`, `tsmatrix`, `tsnames`,
   `build_var_Z`, `companion`. Its own tests, including the `k = 1`
   reduction (§7.14).
3. Implement `fit_var`, `var_select`, `roots`, `is_stable`, `forecast`
   / `predict`.
4. Run the battery. Target ≥ 300 assertions in `test/test_var.jl`.
5. Copy `scripts/*.R` and `scripts/*.py` to `test/verification/var/`
   so every number can be regenerated rather than trusted.
6. `@code_warntype` on the three exported entry points; record the
   benchmark table (§6.8 item 6) in this handoff.
7. Add `src/manual/` chapter on VAR. Do not leave a stub.
8. Mark 10.1 `✅ built` in `development-sequence.md` with Stage-9-level
    detail: what was verified against what, what disagreed, what was
    refused.
9. Record any **new** reference-vs-reference disagreement in the
    charter §5 so 10.2, 10.3, 10.7 and 10.8 inherit it.

---

## 9. Bundle contents

```
fixtures/
  var2_p1.csv  var3_p2.csv  var1_k1.csv  var2_short.csv
  var2_nearunit.csv  var2_explosive.csv
  uschange.csv  econ5.csv
  ref_R.json  ref_py.json  ref_R_extra.json  ref_py_extra.json
scripts/
  gen_fixtures.R    # regenerates every synthetic fixture bit-for-bit
  ref_R.R           # all vars/VARselect reference values -> ref_R.json
  extra2.R          # stability edge cases + R error messages
  ref_py.py         # all statsmodels reference values -> ref_py.json
  cmp.py            # R-vs-Python comparison; reproduces every finding in charter 5
```

`ref_R.json` carries 12 fitted models and 8 selection tables at 17
significant digits; `ref_py.json` carries the statsmodels counterparts.
Between them they hold considerably more than the numbers quoted above —
load them in the test file and assert broadly, rather than transcribing.
