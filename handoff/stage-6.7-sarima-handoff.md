# Handoff: Stage 6.7 — Seasonal ARIMA (Multiplicative Polynomial)

Status: **done.** `fit_sarima`/`SarimaModel` (`src/sarima.jl`,
`test/test_sarima.jl`, `test/test_sarima_bulk.jl`) built per sections
4-6. Every numeric claim re-verified directly against real R and
Python on the exact bundled `sarima_shared.csv`, plus the section 7
"flagged, not yet done" gap closed and a much larger bulk sweep built
beyond this handoff's own single case:

1. **§1-3's `nobs`/coefficient/loglik findings confirmed exactly** on
   `sarima_shared.csv`: `phi=0.3424276441, Phi=0.3242885717,
   loglik=-121.334298, nobs=84` (R) vs `nobs=96` (Python) — same
   architectural split as Stage 6.6, now confirmed for seasonal
   differencing specifically.
2. **§7's explicitly-flagged gap closed**: the handoff's own ground
   truth has `q=Q=0`, so the `theta`/`Theta` parameter-unpacking
   indices (`raw[p+q+1:p+q+P]`, `raw[p+q+P+1:p+q+P+Q]`) were never
   exercised by it. A dedicated synthetic SARIMA(1,0,1)(1,0,1)₄ series
   (all four blocks nonzero) was generated and dual-fit against fresh
   real R and Python — all four coefficients match within 1e-2,
   confirming the indexing arithmetic.
3. **Independently verified beyond the handoff**: `include_mean` is
   silently forced off whenever `d>0` **or** `D>0` — the handoff itself
   only discusses `nobs`'s extension from Stage 6.6, not this
   interaction; confirmed directly against real R that `D>0` alone
   (with `d=0`) also drops the mean (bit-identical coefficients between
   `include.mean=TRUE`/`FALSE`), and confirmed again on a genuinely
   combined `d>0 AND D>0` series.
4. **A real refactor of already-shipped Stage 6.5 code, done carefully**:
   `_hessian_se`/`_opg_se` (previously hardcoded to ARMA's `p`/`q`/
   `include_mean` signature) and the CSS recursion inside
   `_css_start_values` were generalized to take a natural-objective/
   contributions/unpack closure instead — the handoff's own §5 sketch
   assumed these were already generic (`_hessian_se(objective, n0)`),
   which didn't match what Stage 6.5 actually shipped. Full Stage
   6.5/6.6 regression-tested after the refactor, before building this
   stage on top of it.
5. **A 12-case dual R+Python bulk sweep built** (`test/verification/sarima/bulk/`,
   full regeneration pipeline included), well beyond this handoff's
   single case — every polynomial block in isolation and combination,
   `d`/`D` zero and nonzero, seasonal periods 4 and 12. Caught a real
   Windows-filesystem case-insensitivity collision while generating it
   (`P_only.csv` vs `p_only.csv`) — exactly the class of bug this
   project's own header comment on `gaussianssm.jl` already warned
   about — fixed by renaming before it silently overwrote data.

3048+ new SARIMA tests passing (see `development-sequence.md` for the
final count); docs build clean.

---

For a fresh Claude Code session picking this up with no prior context.
This stage gets to lean heavily on work already done: `combined_ar_ma`
is already dual-verified for 20 seasonal cases on the GaussianSSM
branch, and Stage 1.1's `difference` already accepts seasonal
differencing (`D`, `s`) directly. Like 6.6, this should be architected
as thin composition over already-verified pieces, not new fitting logic.

## Where this fits

- **Depends on:** Stage 6.6 (`fit_arima`'s differencing pattern),
  `combined_ar_ma` (GaussianSSM branch, already seasonal-verified),
  Stage 1.1 (`difference(y, d, D, s)` — the seasonal case, not yet
  exercised by 6.6, which only used `D=0`).
- **Parameter ordering — already established, not new**: this project's
  own earlier work confirmed empirically that the optimizer's
  concatenated parameter vector must be ordered `[phi; theta; Phi;
  Theta]` (regular AR, regular MA, seasonal AR, seasonal MA) — cite this
  directly rather than re-deriving it; it's prior, already-verified
  project knowledge.

---

## 1. Verified reference: R `arima(order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=s))`

Same function as 6.5/6.6, `seasonal=` now in scope. Confirmed via a real
fit on a genuine SARIMA(1,0,0)(1,1,0)_12 series (`sarima_shared.csv`,
n=96, 8 years of monthly data):

```r
arima(y, order=c(1,0,0), seasonal=list(order=c(1,1,0), period=12), method="ML")
# coef: phi=0.3424276441  Phi=0.3242885717
# loglik=-121.334298  aic=248.6685961
# nobs=84   <- confirms the pattern from 6.6 extends: nobs = n - d - D*s = 96 - 0 - 1*12 = 84
# se (Hessian): 0.1048658295  0.1105468955
# fit time: 0.0438s
```

## 2. Verified reference: Python `SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s))`

Same series, confirmed:

```python
SARIMAX(y, order=(1,0,0), seasonal_order=(1,1,0,12), trend='n')
# params: ar.L1=0.34243076  seasonal.ar.L12=0.32429148  sigma2=1.03423996
# llf=-121.334298012569  aic=248.668596025138
# nobs=96   <- confirms: full n, NOT n-D*s, same diffuse-augmentation reason as 6.6
# se (OPG): 0.13195281  0.12930198
# fit time: 0.0708s
```

**Every pattern already established in 6.5/6.6 is confirmed to extend
cleanly to the seasonal case**, not just assumed to:
- Coefficients and loglik/AIC match R-to-Python to full precision
  (`-121.334298` both).
- `nobs` discrepancy extends exactly: `84` (R, `n-D*s`) vs `96` (Python,
  full `n`) — same architectural reason as 6.6, now confirmed for
  seasonal differencing specifically, not just regular.
- Standard errors genuinely differ (Hessian 0.105/0.111 vs OPG
  0.132/0.129) — same Hessian-vs-OPG divergence as every prior stage.
- R remains faster (0.044s vs 0.071s), consistent with the established
  pattern, though both take longer than the non-seasonal case (more
  parameters, more complex likelihood surface via `combined_ar_ma`'s
  polynomial multiplication).

---

## 3. Design decision: `nobs = n - d - D*s`, extending 6.6's choice consistently

No new decision needed — this is 6.6's already-made choice (`n-d`,
matching R, matching this project's pre-differencing architecture),
extended by the same logic to `D*s`. Document explicitly that this
still won't match Python's `nobs` (full `n`), same as 6.6.

---

## 4. Proposed Julia API

```julia
fit_sarima(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int};
           include_mean::Bool=true,
           method::Symbol=:ml,
           se_type::Symbol=:hessian,
           optimizer_method::Symbol=:lbfgs,
           start_params::Union{Nothing,Vector{Float64}}=nothing) -> SarimaModel
```

Design notes:
- **Two positional tuples, `order` and `seasonal_order`** — matches
  Python's exact split (`order=`, `seasonal_order=`) rather than R's
  nested `seasonal=list(order=..., period=...)` structure. Python's flat
  4-tuple (`(P,D,Q,s)`) is simpler to work with and matches this
  project's established preference for Python-style naming where the
  two references differ (same reasoning as `adf_test`'s `regression`
  argument).
- **All other keywords identical to 6.5/6.6** — same reasoning as 6.6:
  this stage's job is polynomial composition + seasonal differencing,
  not new fitting options.
- **No `period` as a separate keyword** — folded into `seasonal_order`'s
  4th element, matching Python exactly, avoiding R's separate
  `period=NA` slot that has to stay consistent with the tuple.

### `SarimaModel`

```julia
struct SarimaModel <: UnivariateModel
    phi::Vector{Float64}
    theta::Vector{Float64}
    Phi::Vector{Float64}
    Theta::Vector{Float64}
    mean::Union{Nothing,Float64}
    se::Vector{Float64}           # matches [phi; theta; Phi; Theta] order -- see section 0
    loglik::Float64
    sigma2::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    order::Tuple{Int,Int,Int}
    seasonal_order::Tuple{Int,Int,Int,Int}
    method::Symbol
    se_type::Symbol
    converged::Bool
end
```

---

## 5. Implementation

```julia
"""
    fit_sarima(y, order, seasonal_order; include_mean=true, method=:ml,
               se_type=:hessian, optimizer_method=:lbfgs, start_params=nothing) -> SarimaModel

Fit seasonal ARIMA(p,d,q)(P,D,Q)_s by seasonally-and-regularly
differencing `y` (Stage 1.1's `difference(y, d, D, s)`) and combining
the regular/seasonal AR and MA polynomials via `combined_ar_ma`
(GaussianSSM branch, already dual-verified for 20 seasonal cases) before
handing off to the same `kalman_filter`/optimizer core Stage 6.5 uses.
Parameter vector ordering is `[phi; theta; Phi; Theta]`, per this
project's own earlier empirical verification -- see the Stage 6.7
handoff doc for the dual-verified fitting ground truth this reproduces,
and 6.6's handoff for why `nobs = n - d - D*s`, not Python's full `n`.
"""
function fit_sarima(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int};
                     include_mean::Bool=true, method::Symbol=:ml,
                     se_type::Symbol=:hessian, optimizer_method::Symbol=:lbfgs,
                     start_params::Union{Nothing,Vector{Float64}}=nothing)
    p, d, q = order
    P, D, Q, s = seasonal_order
    yv = tsvalues(y)
    yd = (d > 0 || D > 0) ? difference(yv, d, D, s) : yv
    mu = include_mean ? sum(yd)/length(yd) : 0.0
    yc = yd .- mu

    function objective(raw::Vector)
        phi   = p > 0 ? partrans(raw[1:p]) : Float64[]
        theta = q > 0 ? partrans(raw[p+1:p+q]) : Float64[]
        Phi   = P > 0 ? partrans(raw[p+q+1:p+q+P]) : Float64[]
        Theta = Q > 0 ? partrans(raw[p+q+P+1:p+q+P+Q]) : Float64[]
        ar, ma = combined_ar_ma(phi, Phi, theta, Theta, s)
        ssm = build_statespace(ar, ma)
        loglik, sigma2, v, F, converged = kalman_filter(ssm, yc)
        return converged ? -loglik : 1e10
    end

    x0 = start_params !== nothing ? start_params : zeros(p + q + P + Q)
    result = _optimize(objective, x0; method=optimizer_method)

    n0 = result.minimizer
    phi_hat   = p > 0 ? partrans(n0[1:p]) : Float64[]
    theta_hat = q > 0 ? partrans(n0[p+1:p+q]) : Float64[]
    Phi_hat   = P > 0 ? partrans(n0[p+q+1:p+q+P]) : Float64[]
    Theta_hat = Q > 0 ? partrans(n0[p+q+P+1:p+q+P+Q]) : Float64[]
    ar, ma = combined_ar_ma(phi_hat, Phi_hat, theta_hat, Theta_hat, s)
    ssm = build_statespace(ar, ma)
    loglik, sigma2, = kalman_filter(ssm, yc)

    se = se_type == :hessian ? _hessian_se(objective, n0) : _opg_se(objective, n0, length(yc))

    k = p + q + P + Q + (include_mean ? 1 : 0)
    aic = -2*loglik + 2*k
    bic = -2*loglik + k*log(length(yc))

    return SarimaModel(phi_hat, theta_hat, Phi_hat, Theta_hat, include_mean ? mu : nothing,
                        se, loglik, sigma2, aic, bic, length(yc), order, seasonal_order,
                        method, se_type, result.converged)
end
```

**Note on reuse**: this is deliberately *not* calling `fit_arma`
directly the way 6.6 called it — the seasonal case needs its own
objective function because `combined_ar_ma` has to run *inside* the
optimizer's loop (the combined polynomial changes every iteration as
`phi`/`Phi`/`theta`/`Theta` are searched independently). What *is*
reused directly: `partrans`, `build_statespace`, `kalman_filter`,
`_optimize`, `_hessian_se`/`_opg_se` — every actual numerical primitive,
just recomposed for four parameter blocks instead of two. This is the
right level of reuse, not a violation of the "thin wrapper" principle
from 6.6 — 6.6's wrapper worked because differencing is fully separable
from fitting; seasonal composition genuinely isn't separable from the
optimization loop the same way.

---

## 6. `show`

```julia
function Base.show(io::IO, m::SarimaModel)
    p, q = length(m.phi), length(m.theta)
    P, Q = length(m.Phi), length(m.Theta)
    _, d, _ = m.order
    _, D, _, s = m.seasonal_order
    println(io, "ARIMA(", p, ",", d, ",", q, ")(", P, ",", D, ",", Q, ")[", s, "]",
                 m.mean !== nothing ? " with mean" : "", ", n=", m.nobs,
                 " (", m.method, ", se: ", m.se_type, ")")
    names = vcat(["ar$i" for i in 1:p], ["ma$i" for i in 1:q],
                 ["sar$i" for i in 1:P], ["sma$i" for i in 1:Q],
                 m.mean !== nothing ? ["mean"] : String[])
    coefs = vcat(m.phi, m.theta, m.Phi, m.Theta, m.mean !== nothing ? [m.mean] : Float64[])
    z = coefs ./ m.se
    pval = 2 .* (1 .- _std_normal_cdf.(abs.(z)))
    ct = CoefTable(hcat(coefs, m.se, z, pval), ["Coef.", "Std. Error", "z", "Pr(>|z|)"], names)
    println(io, ct)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "  AIC: ", round(m.aic, digits=2), "  BIC: ", round(m.bic, digits=2))
end
```

---

## 7. Comprehensive test matrix

### Core: dual-verified real fit

| Case | Verified against |
|---|---|
| SARIMA(1,0,0)(1,1,0)_12, n=96 | R: `phi=0.3424276441, Phi=0.3242885717, loglik=-121.334298, aic=248.6685961, nobs=84`; Python: `ar.L1=0.34243076, seasonal.ar.L12=0.32429148, llf=-121.334298012569, aic=248.668596025138, nobs=96` |
| Same case, `se_type=:hessian` vs `:opg` | R se=[0.1049, 0.1105] vs Python se=[0.1320, 0.1293] — genuinely different, same assertion pattern as 6.5/6.6 |
| `P=Q=D=0` reduces exactly to 6.6's `fit_arima` | Structural regression guard, same role as 6.6's `d=0` test |
| Full SARMA(1,1)(1,1)_s (all four polynomial blocks nonzero) | Not yet verified numerically here — worth a dedicated R/Python fit before trusting the parameter-unpacking indices (`p+q+1:p+q+P` etc.) in section 5, since that's the highest-risk new arithmetic in this stage |

```julia
using Test, DelimitedFiles

@testset "fit_sarima — dual-verified" begin
    y = vec(readdlm("sarima_shared.csv", ',', skipstart=1))

    m = fit_sarima(y, (1,0,0), (1,1,0,12); include_mean=false, method=:ml)
    @test m.converged
    @test isapprox(m.phi[1], 0.3424276441; atol=1e-3)
    @test isapprox(m.Phi[1], 0.3242885717; atol=1e-3)
    @test isapprox(m.loglik, -121.334298; atol=1e-2)
    @test m.nobs == length(y) - 1*12   # 84, matching R's convention

    m_hess = fit_sarima(y, (1,0,0), (1,1,0,12); include_mean=false, se_type=:hessian)
    m_opg  = fit_sarima(y, (1,0,0), (1,1,0,12); include_mean=false, se_type=:opg)
    @test !isapprox(m_hess.se, m_opg.se; atol=1e-3)

    # no seasonal terms -> must match fit_arima's ARIMA(1,0,0) exactly
    m_nonseasonal = fit_sarima(y, (1,0,0), (0,0,0,12); include_mean=false)
    m_direct = fit_arima(y, (1,0,0); include_mean=false)
    @test isapprox(m_nonseasonal.phi, m_direct.arma.ar; atol=1e-8)

    @test_throws ArgumentError fit_sarima(y, (1,0,0), (1,1,0,12); method=:bogus)
end
```

**Flagged, not yet done**: a case exercising all four polynomial blocks
(p,q,P,Q all > 0) needs its own dedicated R/Python ground truth before
trusting the index arithmetic in section 5's `objective` function — the
single seasonal case verified here has `q=Q=0`, so it doesn't actually
exercise the `theta`/`Theta` slicing at all. This is exactly the kind of
gap that's bitten this project before (an untested code path looking
fine until the specific combination that exercises it).

---

## 8. Performance

Same story as 6.6 — no new performance surface beyond what 6.5 already
established, just a larger parameter count (up to 4 blocks instead of
1-2) feeding the same `_optimize`/`kalman_filter` loop. Real reference
timings from section 1-2: R 0.044s, Python 0.071s, for a 2-parameter
seasonal fit on n=96 — both slower than the non-seasonal ARMA(1,1) case
from 6.5 (R: 2-8ms), consistent with a more complex likelihood surface,
not a red flag. Profile once running; same "don't pre-optimize without a
profile" discipline as every earlier stage.

---

## 9. What to do with this

1. Implement `fit_sarima`/`SarimaModel` per sections 4-6.
2. Run the tests in section 7, **specifically prioritizing a
   dedicated ground-truth case with q,Q > 0** before trusting the
   parameter-unpacking indices — this is the highest-risk new code in
   this stage.
3. Confirm `combined_ar_ma`'s already-established seasonal verification
   (GaussianSSM branch, 20 cases) still holds when called *inside* the
   optimizer loop, not just at fixed known coefficients — a subtly
   different usage pattern worth a specific check.
4. Update `development-sequence.md`'s Stage 6.7 row: mark implemented,
   note the `nobs = n - d - D*s` convention (extending 6.6's decision),
   and flag the untested `q,Q>0` gap explicitly if it isn't closed
   before shipping.
