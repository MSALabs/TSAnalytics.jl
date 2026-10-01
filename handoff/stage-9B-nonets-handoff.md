# Handoff: Stage 9B — Everything Else

## Status: TIER 1 COMPLETE, 2.1 COMPLETE (2026-10-01)

**§2.1 (seasonal unit-root test) is done**, and two of its premises were
wrong:

- **R's `nsdiffs` does not default to Canova-Hansen.** It defaults to
  `test="seas"`, an STL seasonal-strength heuristic with a `0.64`
  threshold. CH and HEGY need the separate `uroot` package and are not
  reachable by default. Implemented R's real default plus OCSB.
- **CRAN being unreachable did not make this single-verified.** R's
  `forecast` 9.0.2 was already installed here, so everything was read
  from R's source and executed. `ocsb_test` follows **R**, not
  `pmdarima`, which differs from R on three counts with measured gaps up
  to `1.25`.

`auto_arima(y; seasonal=true, m=12)` is now fully automatic. Tier 2
continues at 2.2 (`vcov` coverage).


1.1, 1.2, 1.3 and 1.4 are all implemented, tested and documented. Every
reference value in this document was **re-executed** against real R
4.6.0 / scipy this session rather than transcribed, and all of them
reproduced exactly. Tier 2 starts at 2.1 (the seasonal unit-root test).

Corrections to this document, found by verifying its premises:

- **The StatsAPI table understates two rows.** `ArimaxModel` and
  `SarimaxModel` are shown as entirely empty; they already had `coef`,
  `loglikelihood`, `aic`, `bic` and `nobs`. Only `vcov`, `residuals`
  and `predict` were missing, so Tier 1 was smaller than scoped.
  (`stderror` exists only on `ARXModel` and is absent from the table.)
- **§2.2's ARIMAX `se` target is in R's order** `[ar1, intercept, x]`.
  Julia orders `[exog..., arma...]`, so the correct assertion is
  `[0.1110204142, 0.3269969227, 0.0686772108]`.
- **`beta` already carries the intercept** as its last entry, with
  `arma.mean` left `nothing` — `fit_arimax` appends a column of ones to
  the design matrix. Now asserted, since the name invites the wrong
  reading.
- **§1.4's `mstl(lambda=0.0)` item was already done.** Verified:
  `lambda=0.0` differs from `nothing` and matches `1e-8`. Stage 3.3
  already recorded the falsy-zero divergence deliberately. The test is
  a regression guard, not new work.
- **The Bartlett line numbers are wrong** — `unitroot.jl:659` (KPSS)
  and `:799` (PP), not 475/615. And the two were *not* a literal copy:
  KPSS divided each autocovariance by `n` inside the loop, PP summed
  raw and divided at the end. Algebraically identical, so the shared
  `_bartlett_lrv` is checked bit-for-bit against the pre-refactor
  statistics.
- **§1.4's `pacf` finding is half right.** R's `pacf()` does ignore its
  `method` argument — confirmed. But the claim that R's answer "matches
  none of the three exactly" is wrong: **R equals this package's
  `:ywm` exactly**, while `:yw` and `:ols` equal `statsmodels`' own.
  So `:burg` follows `statsmodels` because R has no Burg PACF at all,
  not because the conventions are unresolved.
- **`stl(seasonal_window=:periodic)` needed `inner=2` to match R.**
  R defaults to `inner=2`; this package defaults to `5`, matching
  `statsmodels` — a pre-existing documented divergence, not a new one.
  With `inner=2` Julia reproduces R's `s.jump=1` run to 1e-8. R's own
  periodic branch uses `s.jump=121` and so differs from itself by
  ~2e-6; this package has no jump shortcut and lands on the unjumped
  answer.
- **`lambda=:auto` had a reference choice to make.** `scipy.stats.boxcox`
  is unbounded and returns `-1.836` on the shipped fixture, which fails
  this document's own `-1 <= lambda <= 2` assertion. Implemented bounded
  on `[-1, 2]`, matching R's `BoxCox.lambda` and this package's own
  `guerrero_lambda`; widening the bounds reproduces scipy to 1e-5.
  `MSTLDecomposition` gained a `lambda` field to report what was used.

Also closed, beyond the written scope: `ArmaModel` had no `predict` at
all, so it gained one (delegating to the `ArimaModel` `d=0` path), and
`runtests.jl` now imports the StatsAPI accessors once so a test cannot
pass in isolation and error under the full suite.

The original plan follows, unchanged.

---

Tiered into **immediate**, **immediate next** and **later**. Every
reference value below was generated this session; fixtures ship in
`fixtures/`.

---

## Verified StatsAPI coverage, commit `113e231`

Read from source, not from notes. `residuals` has landed for three
types since the last audit.

| Model | coef | vcov | residuals | predict | loglik | aic | bic | nobs |
|---|---|---|---|---|---|---|---|---|
| `ARXModel` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ArimaModel` | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SarimaModel` | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GarchModel` | ✓ | — | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| `AutoregGarchModel` | ✓ | — | — | — | ✓ | ✓ | ✓ | ✓ |
| `ArmaModel` | — | — | ✓ | — | — | — | — | — |
| `ArimaxModel` | — | — | — | — | — | — | — | — |
| `SarimaxModel` | — | — | — | — | — | — | — | — |
| `ExponentialSmoothingModel` | — | — | — | — | — | — | — | — |

### The root cause

```
ArmaModel      ar ma mean se loglik aic bic nobs order method se_type converged   <- no series
SarimaModel    phi theta Phi Theta mean se loglik ... converged                   <- no series
ArimaModel     arma d original_y                                                  <- has series
ArimaxModel    ... arma d original_y exog se loglik ... converged                 <- has both
SarimaxModel   ... arma seasonal_order original_y exog se loglik ... converged     <- has both
```

**`ArmaModel` and `SarimaModel` discard the series.** One fact, three
symptoms: the inconsistent `predict(m::SarimaModel, y, horizon)`
signature, `diagnostic_plot` needing residuals passed explicitly, and
`residuals()` having been a per-type job.

**`ArimaxModel` and `SarimaxModel` retain `original_y` *and* `exog`** —
they already hold everything forecasting needs.

---

# TIER 1 — IMMEDIATE

Finish what is started. No new statistics.

## 1.1 `predict`/`forecast` for `ArimaxModel` and `SarimaxModel`

The most consequential gap on either list. Forecasting with exogenous
regressors is largely the reason to fit ARIMAX, and Stage 8 ships
models that fit, report coefficients and cannot forecast.

### Reference — R and Python, dual-verified

`fixtures/ax_y.csv` (140 obs), `ax_x.csv` (140), `ax_xf.csv` (10 future
regressor values). AR(1) with `phi = 0.6`, true `beta = 2.0`, seed 31.

```
R:  arima(y, order=c(1,0,0), xreg=x)
      ar1        0.5847256127   se 0.0686772108
      intercept -0.3070410373   se 0.3269969228
      x          2.1364359236   se 0.1110204142
      loglik  -197.959062542    aic 403.918125084
    predict(m, n.ahead=10, newxreg=xf)
      forecast[1:5]  -4.4532060919 -3.6415736308 -3.5780284626 -3.1939196254 -2.8211025062
      se[1:5]         0.9935811581  1.1509701955  1.2000561975  1.2163845829  1.2219172707

Python: ARIMA(y, exog=x, order=(1,0,0), trend='c')
      ar.L1   0.5847276135    const -0.3070208392    x1 2.1364259403
      loglik -197.95906255    aic 403.918125
      forecast[1:5]  -4.45319474 -3.64155617 -3.57800476 -3.19389345 -2.82107532
      se[1:5]         0.99357507  1.15096415  1.20005047  1.21637907  1.22191186
```

**They agree to 5–6 decimals.** That is a properly dual-verified target.

### Design

Match `ARXModel`'s `predict` signature — it has already answered the
future-regressor question; do not invent a second convention.

```julia
predict(m::ArimaxModel,  newexog, horizon::Integer; level=[80.0, 95.0]) -> Forecast
predict(m::SarimaxModel, newexog, horizon::Integer; level=[80.0, 95.0]) -> Forecast
```

### Tests

```julia
@testset "ARIMAX forecast matches R and Python" begin
    y  = readdlm("fixtures/ax_y.csv")[:,1]
    x  = readdlm("fixtures/ax_x.csv")[:,1]
    xf = readdlm("fixtures/ax_xf.csv")[:,1]
    m  = fit_arimax(y, reshape(x,:,1); order=(1,0,0))

    @test isapprox(m.beta[1], 2.1364359236; atol=1e-3)
    @test isapprox(m.loglik, -197.959062542; atol=1e-4)

    f = forecast(m, reshape(xf,:,1), 10)
    @test isapprox(f.point[1:5],
        [-4.4532060919, -3.6415736308, -3.5780284626, -3.1939196254, -2.8211025062];
        atol=1e-3)
    @test isapprox(f.se[1:5],
        [0.9935811581, 1.1509701955, 1.2000561975, 1.2163845829, 1.2219172707];
        atol=1e-3)
end

@testset "ARIMAX forecast: newexog validation" begin
    m  = fit_arimax(y, reshape(x,:,1); order=(1,0,0))
    @test_throws ArgumentError forecast(m, reshape(xf[1:5],:,1), 10)   # too few rows
    @test_throws ArgumentError forecast(m, hcat(xf, xf), 10)           # wrong column count
    @test_throws ArgumentError forecast(m, reshape(xf,:,1), 0)         # zero horizon
end

@testset "ARIMAX se grows with horizon and converges" begin
    f = forecast(m, reshape(xf,:,1), 10)
    @test issorted(f.se)
    @test f.se[10] - f.se[9] < f.se[2] - f.se[1]    # growth decelerates for stationary AR
end

@testset "zero-coefficient exog reduces to plain ARIMA" begin
    # exog that is identically zero must give the ARIMA forecast
    z = zeros(length(y), 1)
    a = fit_arimax(y, z; order=(1,0,0))
    b = fit_arima(y; order=(1,0,0))
    @test isapprox(a.loglik, b.loglik; atol=1e-6)
end
```

The error messages must name the mismatch. Chapter 36's X-13 work hit
exactly this failure — a regressor stopping at the end of the sample —
and the binary's message was clear enough to save an afternoon.

---

## 1.2 Struct fix, uniform signature, `predict`/`forecast` alias

**Do this *with* 1.1**, not after — otherwise the ARIMAX signature gets
designed twice.

**Add `original_y` to `ArmaModel` and `SarimaModel`.** That removes the
third positional argument, lets `diagnostic_plot` take the model alone,
and makes the alias uniform rather than type-dependent.

**The alias.** `StatsAPI.predict` is canonical; `forecast` forwards
mechanically, one method per type:

```julia
forecast(m::T, args...; kwargs...) = StatsAPI.predict(m, args...; kwargs...)
```

Both docstrings state they are aliases and that neither is deprecated —
users from R expect `forecast`, users from the Julia statistics
ecosystem expect `predict`.

```julia
@testset "predict and forecast are exact aliases on every type" begin
    for m in (arma_fit, arima_fit, sarima_fit, arimax_fit, sarimax_fit, arx_fit)
        a = predict(m, 6); b = forecast(m, 6)
        @test a.point == b.point
        @test a.se    == b.se
        @test a.lower == b.lower && a.upper == b.upper
    end
end

@testset "signature is uniform — no third positional argument anywhere" begin
    @test applicable(predict, sarima_fit, 6)
    @test applicable(predict, arima_fit,  6)
    @test applicable(predict, arma_fit,   6)
end

@testset "diagnostic_plot takes the model alone" begin
    @test diagnostic_plot(sarima_fit) isa DiagnosticPlotResult
end
```

**Grep for callers of the three-argument form before changing it** —
Manual pages and Introduction chapters very likely use it, and the docs
need updating in step.

---

## 1.3 `ccf` — cross-correlation function

The one genuinely new finding from the six-book sweep. Present in all
six, absent from the package, and `ACFResult` already exists to carry
it. Chapter 5's handoff calls the `soi`/`rec` lagged relationship *"the
whole motivation for cross-correlation"*.

### A real convention disagreement

Fixture `fixtures/ccf_x.csv`, `ccf_y.csv` — 145 obs, `y` lags `x` by 3.

```
R  ccf(x, y, lag.max=6)   -- TWO-SIDED
   -6 +0.2661575938   -5 +0.4159572432   -4 +0.6633372049
   -3 +0.9636929430   <- peak
   -2 +0.6817189202   -1 +0.4641562350    0 +0.3365029144
   +1 +0.2255713144   +2 +0.1562916927   +3 +0.0941916568
   +4 +0.0907835083   +5 +0.1170574662   +6 +0.1431579617

statsmodels ccf(x, y)     -- NON-NEGATIVE LAGS ONLY
    0 +0.3365029144   +1 +0.2255713144   ... +6 +0.1431579617
statsmodels ccf(y, x)
    0 +0.3365029144   +3 +0.9636929430   <- the peak, reached by swapping arguments
```

**`statsmodels` returns one-sided output, so `ccf(x, y)` does not
contain the peak at all.** A user porting from R who calls it and looks
for the maximum will not find it. Values match exactly where they
overlap: R's lag −k equals `statsmodels` `ccf(y, x)` at lag +k.

**Implement R's two-sided convention** — it is what all six books use
and what is actually useful. Document the Python difference in the
docstring.

```julia
@testset "ccf matches R's two-sided convention" begin
    x = readdlm("fixtures/ccf_x.csv")[:,1]
    y = readdlm("fixtures/ccf_y.csv")[:,1]
    r = ccf(x, y, 6)
    @test r.lags == -6:6
    @test isapprox(r.values[r.lags .== -3][1], 0.9636929430; atol=1e-8)
    @test isapprox(r.values[r.lags .==  0][1], 0.3365029144; atol=1e-8)
    @test isapprox(r.values[r.lags .== +3][1], 0.0941916568; atol=1e-8)
    @test argmax(r.values) == findfirst(==(-3), r.lags)   # peak identifies the lead
end

@testset "ccf(x,y) at lag -k equals ccf(y,x) at lag +k" begin
    a = ccf(x, y, 6); b = ccf(y, x, 6)
    @test isapprox(a.values, reverse(b.values); atol=1e-12)
end

@testset "ccf of a series with itself is the acf" begin
    @test isapprox(ccf(x, x, 6).values[7:end], acf(x, 0:6).values; atol=1e-10)
end

@testset "ccf edge cases" begin
    @test_throws DimensionMismatch ccf(randn(50), randn(60), 5)
    @test_throws ArgumentError ccf(x, y, 0)
    @test_throws ArgumentError ccf(x, y, length(x))     # lag.max too large
end
```

Return an `ACFResult` tagged `kind = :ccf` so the existing plot recipe
serves it with a changed axis label — the same mechanism `:acf` and
`:pacf` already share.

---

## 1.4 Option completions

Three arguments that are accepted and then rejected. Small, no design
decisions.

### `pacf(method=:burg)` — and a finding about the reference

```
R pacf(y, lag.max=6, method=...)   -- ALL FOUR METHODS IDENTICAL
  yule-walker  0.9202807457  0.0792418029  0.4684062110  0.2150550916 -0.4616234266  0.0163384787
  burg         (identical)
  ols          (identical)
  mle          (identical)
```

**R's `pacf()` ignores its `method` argument.** Burg lives in
`ar.burg()`, not `pacf()`. So R is *not* the reference here.

```
statsmodels, lags 1-6 -- three genuinely different answers
  pacf_yw    0.92801420  0.08932240  0.53211610  0.29004921 -0.52704150  0.00687497
  pacf_ols   0.97182246  0.09673555  0.78073061  0.87902996 -0.63856087 -0.32056007
  pacf_burg  0.94515892  0.08367719  0.73939568  0.68095041 -0.83492890 -0.38198286
```

R's single answer matches **none** of the three exactly — a denominator
convention difference worth pinning down.

```julia
@testset "pacf(:burg) matches statsmodels pacf_burg" begin
    y = readdlm("fixtures/ets_y.csv")[:,1]
    p = pacf(y, 1:6; method=:burg)
    @test isapprox(p.values,
        [0.94515892, 0.08367719, 0.73939568, 0.68095041, -0.83492890, -0.38198286];
        atol=1e-6)
end

@testset "pacf methods genuinely differ" begin
    y = readdlm("fixtures/ets_y.csv")[:,1]
    @test !isapprox(pacf(y,1:6;method=:burg).values, pacf(y,1:6;method=:yw).values; atol=1e-3)
    @test !isapprox(pacf(y,1:6;method=:burg).values, pacf(y,1:6;method=:ols).values; atol=1e-3)
end

@testset "burg pacf is bounded" begin
    @test all(abs.(pacf(randn(200), 1:20; method=:burg).values) .<= 1.0)
end
```

**First check which convention the existing `:yw` follows** — R's or
`statsmodels`'. If it tracks R, note in the docstring that `:burg`
necessarily follows `statsmodels` because R has no Burg PACF.

### `stl(seasonal_window=:periodic)` and `mstl(lambda=:auto)`

```julia
@testset "stl periodic gives a bit-identical seasonal pattern each cycle" begin
    y = readdlm("fixtures/ets_y.csv")[:,1]
    d = stl_decompose(y, 4; seasonal_window=:periodic)
    s = d.seasonal
    @test s[1:4] == s[5:8] == s[9:12]        # exact, not approximate
end

@testset "mstl lambda=:auto selects and applies a transform" begin
    y = abs.(readdlm("fixtures/ets_y.csv")[:,1]) .+ 10
    a = mstl(y, [4]; lambda=:auto)
    @test a.lambda isa Float64 && -1 <= a.lambda <= 2
end

@testset "mstl lambda=0 applies a LOG, not no transform" begin
    # statsmodels has a live falsy-zero bug here: `elif self.lmbda:` skips
    # the branch at zero, so lmbda=0 silently means no transform.
    # Julia must check `!== nothing`.
    y = abs.(readdlm("fixtures/ets_y.csv")[:,1]) .+ 10
    z = mstl(y, [4]; lambda=0.0)
    n = mstl(y, [4]; lambda=nothing)
    e = mstl(y, [4]; lambda=1e-8)
    @test !isapprox(z.trend, n.trend; atol=1e-6)   # must NOT equal no-transform
    @test isapprox(z.trend, e.trend; rtol=1e-4)    # must equal near-zero lambda
end
```

That last test encodes a bug verified live in `statsmodels` this
project — `lmbda=0` there produces output bit-identical to no transform
(trend[0] 11.285054 both) while `lmbda=1e-8` gives 2.391686. Julia must
not reproduce it.

### Housekeeping

Factor out the Bartlett long-run-variance kernel, written twice inline
at `unitroot.jl:475` (`kpss_test`) and `:615` (`pp_test`). Fix the
stale Stage 6 note at `abstract.jl:21`.

```julia
@testset "shared Bartlett kernel leaves both tests unchanged" begin
    y = readdlm("fixtures/ets_y.csv")[:,1]
    @test isapprox(kpss_test(y).statistic, KPSS_BEFORE_REFACTOR; atol=1e-12)
    @test isapprox(pp_test(y).statistic,   PP_BEFORE_REFACTOR;   atol=1e-12)
end
```

Capture both values **before** refactoring.

---

# TIER 2 — IMMEDIATE NEXT

Contained new work with a clear reference.

## 2.1 Seasonal unit-root test

The only reason `auto_arima` is not fully automatic — it detects `d` by
repeated `kpss_test` but cannot detect `D`. Not a numbered roadmap row;
a noted gap inside Stage 6.8, **documented in three separate places**
while writing the Manual.

Implement **OCSB** (what `pmdarima` uses) and/or **Canova-Hansen**
(R's `forecast::nsdiffs`). `pmdarima` is installable and is the
reachable reference; CRAN is not.

```julia
ocsb_test(y, m; maxlag=nothing) -> OCSBTest
nsdiffs(y, m; test=:ocsb, max_D=1) -> Int
```

```julia
@testset "nsdiffs finds D=1 on strongly seasonal data" begin
    t = 1:120
    y = 100 .+ 0.3 .* t .+ 10 .* sin.(2π .* t ./ 4) .+ randn(120)
    @test nsdiffs(y, 4) == 1
end

@testset "nsdiffs finds D=0 on non-seasonal data" begin
    @test nsdiffs(cumsum(randn(120)), 4) == 0
end

@testset "auto_arima selects D without being told" begin
    y = readdlm("fixtures/ets_y.csv")[:,1]
    m = auto_arima(y; seasonal=true, m=4)          # no D= passed
    @test m.seasonal_order[2] >= 1
end

@testset "explicit D still overrides" begin
    m = auto_arima(y; seasonal=true, m=4, D=0)
    @test m.seasonal_order[2] == 0
end
```

Update the three places documenting the limitation once this lands.

## 2.2 `vcov` coverage

**Verified as a fit-path change, not an accessor.** `ARXModel` stores a
full `vcov` matrix — hence its one-line accessor. Everything else
stores `se::Vector{Float64}`, the diagonal only; `ArimaModel` has no
`se` field at all. **`vcov` cannot be reconstructed from what is
retained** — the fit functions must keep the full matrix.

Scope it before starting. Touches estimation code across several model
families.

```julia
@testset "vcov is consistent with stderror everywhere" begin
    for m in (arma_fit, arima_fit, sarima_fit, garch_fit, arimax_fit)
        @test size(vcov(m)) == (length(coef(m)), length(coef(m)))
        @test isapprox(sqrt.(diag(vcov(m))), stderror(m); rtol=1e-10)
        @test issymmetric(vcov(m))
        @test all(diag(vcov(m)) .>= 0)
    end
end

@testset "ARIMAX vcov diagonal matches the verified R standard errors" begin
    m = fit_arimax(y, reshape(x,:,1); order=(1,0,0))
    se = sqrt.(diag(vcov(m)))
    @test isapprox(se, [0.0686772108, 0.3269969228, 0.1110204142]; atol=1e-3)
end
```

That last test has a real target — R's `se` from section 1.1.

## 2.3 Distributional forecast accuracy

CRPS, pinball loss, Winkler score, empirical coverage. **Every
prediction interval the package emits is currently unfalsifiable** —
`accuracy` covers point forecasts only. fpp3 §5.9 specifies all of
them.

```julia
@testset "pinball loss at the median equals half the MAE" begin
    y = randn(100); q = randn(100)
    @test isapprox(pinball_loss(y, q, 0.5), 0.5*mean(abs.(y .- q)); rtol=1e-10)
end

@testset "CRPS of a point forecast equals the absolute error" begin
    # a degenerate predictive distribution reduces CRPS to |y - mu|
    @test isapprox(crps_normal(3.0, 2.5, 1e-10), abs(3.0-2.5); atol=1e-6)
end

@testset "coverage approaches nominal on well-specified simulated data" begin
    hits = 0; N = 400
    for _ in 1:N
        y = simulate_known_process(80)
        f = forecast(fit_arima(y[1:79]; order=(1,0,0)), 1)
        hits += (f.lower[1,2] <= y[80] <= f.upper[1,2])
    end
    @test 0.90 <= hits/N <= 0.99      # nominal 95%
end
```

## 2.4 The rest of Tier 2

- **Bias-corrected back-transformation.** Verified *not* a silent bug —
  no forecast path auto-transforms. But the book teaches
  log-then-forecast, so a corrected variant should exist.
  Test: `boxcox_inv` exact on a round trip; the corrected variant
  strictly larger on a skewed series.
- **Fourier / harmonic seasonal terms.** ITSR §5.6; fpp3 leans on them
  where `m` is large. Test: `K` pairs give `2K` columns, orthogonality,
  and reproduction of seasonal dummies at `K = m/2`.
- **AR / parametric spectral estimation.** S&S §4.5, ITSR §9.9, R's
  `spec.ar`. Test against a known AR(2) spectral peak.
- **Forecast combination.** Montgomery §7.5. Test: equal weights equal
  the simple mean; optimal weights never worse in-sample.
- **GARCH `dist=:t`.** Currently throws with a clear documented message
  — correct behaviour, real limitation. Test: as `ν → ∞` the
  log-likelihood approaches the normal case.
- **Missing-data policy**, `include_mean` search, classical AutoReg
  (scope honestly as SAS `PROC AUTOREG` parity, not R/Python catch-up —
  the full ML + AR-GARCH tier already exists as Stage 8.5).

---

# TIER 3 — LATER

| Group | Items |
|---|---|
| **Dynamic regression** | Transfer function models (b,r,s response), intervention analysis. Montgomery ch. 6, S&S §5.5, ITSR ch. 10 — three books give it chapter-level treatment, and chapters 32/35 already reference "intervention terms" the package lacks |
| **VAR track** | VAR, Granger, IRF, FEVD, SVAR |
| **Cointegration** | Engle-Granger, Johansen, VECM |
| **State-space extensions** | UCM (first real consumer of diffuse init), dynamic factor, bootstrapping |
| **Nonlinear** | Markov-switching, TAR, STAR, nonlinearity tests |
| **Long memory** | ARFIMA, fractional differencing. S&S §5.1, ITSR ch. 8 |
| **Frequency domain** | Coherence, cross-spectra, phase — natural once `ccf` and spectral exist |
| **Volatility** | Multivariate GARCH/DCC, stochastic volatility |
| **Forecasting** | Hierarchical reconciliation (largest single gap against fpp3), quantile forecasting, count data, change-point, ML reduction |
| **Regression** | HAC/Newey-West, rank-deficient handling, public-OLS decision |
| **Finance** | VaR, expected shortfall, extreme value theory — arguably a separate package |

---

## Performance

**`ccf`.** The naive double loop is `O(n·k)`; for large `k` use the FFT
route the periodogram already depends on. Benchmark both and pick by
`k` — crossover is usually around `k ≈ 30`. Demean once, outside the
lag loop, not inside it.

**ARIMAX forecast.** The recursion is sequential. Preallocate the
output `Forecast` buffers once; use `@views` for the exog slices rather
than copying rows. `mul!` for the `X·beta` product instead of `X*beta`,
which allocates.

**`vcov`.** Retaining a full matrix per fit costs `O(k²)` memory where
`k` is small — negligible. Do not be tempted to recompute it lazily on
access; the information needed is cheapest at fit time.

**`nsdiffs`.** Called inside `auto_arima`'s search. It fits a
regression per candidate lag, so cache the differenced series across
calls rather than recomputing.

**General, applies throughout:**

- `@code_warntype` on every new hot path. A `Union{Nothing, T}` keyword
  reaching an inner loop poisons inference — resolve it at the
  boundary.
- Preallocate outside optimiser loops; the objective is called
  hundreds of times.
- `@views` on slices in loops; `@inbounds` once indices are provably
  valid.
- Benchmark allocations, not only time. A hot loop should report zero.
- Add each new function to the existing benchmark suite with a recorded
  baseline, so a later change that regresses it is visible.

---

## Order of work

1. **1.1 and 1.2 together** — one piece of work; splitting them means
   designing the ARIMAX signature twice.
2. **1.3 `ccf`** — independent, can run in parallel.
3. **1.4 option completions and housekeeping** — capture the
   before-refactor values first.
4. Then Tier 2, starting with **2.1 seasonal unit-root test**.

---

## Checklist — Tier 1

- [ ] `predict`/`forecast` for `ArimaxModel` and `SarimaxModel`, matching
      the R/Python reference to `atol=1e-3`
- [ ] `newexog` validation errors name the mismatch
- [ ] `original_y` added to `ArmaModel` and `SarimaModel`
- [ ] Three-positional-argument `predict` removed; **callers grepped and
      updated**, docs included
- [ ] `forecast` aliases `predict` on every forecastable type,
      mechanically, both docstrings saying so
- [ ] `ccf` two-sided, R convention, `ACFResult` with `kind = :ccf`
- [ ] `ccf` Python one-sided difference documented
- [ ] `pacf(:burg)` matching `statsmodels`; the R-ignores-`method`
      finding recorded in the docstring
- [ ] `stl(:periodic)` bit-identical across cycles
- [ ] `mstl(lambda=0.0)` applies a log — the falsy-zero test passes
- [ ] Bartlett kernel factored out, both statistics unchanged to 1e-12
- [ ] `@code_warntype` clean on every new hot path; allocations recorded
- [ ] README StatsAPI coverage table updated to match reality
- [ ] British-Indian spelling in docstrings
