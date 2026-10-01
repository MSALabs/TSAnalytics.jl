# Handoff: Stage 9A — Linear ETS

The ETS family in innovations state space form, scoped to the linear
subset. Every reference value below was generated this session and the
fixtures ship alongside this document in `fixtures/`.

---

## 1. Scope — six models, not thirty

Full taxonomy: ETS(E,T,S), Error ∈ {A, M}, Trend ∈ {N, A, Ad, M, Md},
Seasonal ∈ {N, A, M}. Thirty combinations.

**This stage takes the linear subset**: additive error, trend ∈ {N, A,
Ad}, seasonal ∈ {N, A}.

| Model | Equivalent | Status |
|---|---|---|
| ETS(A,N,N) | simple exponential smoothing | `holt_winters` has it |
| ETS(A,A,N) | Holt's linear | has it |
| ETS(A,N,A) | seasonal, no trend | has it |
| ETS(A,A,A) | additive Holt-Winters | has it |
| **ETS(A,Ad,N)** | damped Holt | **new** |
| **ETS(A,Ad,A)** | damped additive Holt-Winters | **new** |

The additive family is exactly the subset expressible as a linear
Gaussian state space model, so it runs on the `GaussianSSM` engine
already built — the README's second design principle names this
explicitly. Multiplicative error and seasonal are the non-linear half,
need simulation-based intervals, and are correctly out of scope.

**What ETS adds over `holt_winters`:** a likelihood (hence AIC/AICc/
BIC), automatic selection, analytic prediction intervals, and the
damped trend parameter φ. `ExponentialSmoothingModel` today carries
`alpha`, `beta`, `gamma`, the state path, `fitted`, `resid`, `sse` —
and no `loglikelihood`, no `aic`, no `phi`.

---

## 2. The fixture

`fixtures/ets_y.csv` — **120 observations, quarterly (m = 4), simulated
from a true ETS(A,A,A) innovations process**:

```
alpha = 0.40   beta = 0.10   gamma = 0.30   sigma = 2.0
initial level 100.0, initial trend 0.35, seasonal [8.0, -3.0, -6.0, 1.0]
seed 20260101
```

**This fixture was built deliberately.** A first attempt used a
deterministic trend plus fixed seasonal plus noise; every smoothing
parameter optimised to its lower bound (0.0001, 1e-08) and φ to its
upper bound (0.98). Parameters pinned at bounds do not discriminate a
correct implementation from a broken one. Simulating from a genuine ETS
process puts them in the interior, where the optimiser is actually
tested.

First eight values: `106.287284, 99.590412, 91.869762, 99.340118,
103.019984, 93.897414, 86.069331, 96.029118`

---

## 3. Reference values — `statsmodels` 0.15.0

All six linear models, `error='add'`, on the fixture above. Full output
in `fixtures/ets_ref.json`.

| Model | loglik | AICc | SSE | k |
|---|---|---|---|---|
| ETS(A,N,N) | −407.564164 | 821.33522 | 6262.60017 | 2 |
| ETS(A,A,N) | −394.328651 | 799.18362 | 5022.88055 | 4 |
| ETS(A,N,A) | −268.608601 | 554.51450 | 617.96059 | 7 |
| **ETS(A,A,A)** | **−245.248153** | **512.51465** | **418.67036** | 9 |
| ETS(A,Ad,N) | −394.588185 | 801.91973 | 5044.65441 | 5 |
| ETS(A,Ad,A) | −245.483300 | 515.41104 | 420.31440 | 10 |

Fitted smoothing parameters:

```
ETS(A,N,N)    level 0.354833
ETS(A,A,N)    level 0.134426  trend 0.019177
ETS(A,N,A)    level 0.715532                    seasonal 0.284439
ETS(A,A,A)    level 0.414046  trend 0.075884    seasonal 0.363091
ETS(A,Ad,N)   level 0.134640  trend 0.031432                        phi 0.98
ETS(A,Ad,A)   level 0.363127  trend 0.119297    seasonal 0.338276   phi 0.98
```

**Two things worth noting.** ETS(A,A,A) recovers α = 0.414, β = 0.076,
γ = 0.363 against a true 0.40 / 0.10 / 0.30 — close, and not exact,
which is what finite-sample estimation looks like. And **AICc selects
ETS(A,A,A), the true data-generating process**, which is a genuine
end-to-end test of the whole selection machinery rather than of any one
component.

**φ = 0.98 in both damped fits is `statsmodels`' upper bound**, not an
interior optimum. If a Julia fit returns φ in the interior on the same
data, that is a specification difference worth investigating rather
than an error — check the bound convention before assuming either is
wrong.

---

## 4. Reference values — base R `HoltWinters`

`fixtures/hw.R`. Base `stats`, no CRAN needed. **Fixed** parameters, so
this isolates the recursion from the optimiser:

```
HoltWinters(ts(y, frequency=4), alpha=0.4, beta=0.1, gamma=0.3, seasonal="additive")

SSE               496.087127186
coef a, b         166.9443758983   1.6728034779
coef s1..s4       9.4353561644  3.5767146876  -4.2667419403  4.4179122017
fitted xhat[1:4]  105.6007048823  95.2291855143  87.5643192297  94.0880570138
forecast[1:4]     178.052535541  173.866697542  167.696044392  178.053502012
```

`forecast::ets()` could not be obtained — **CRAN has been unreachable
throughout this project** and a fresh install attempt failed again this
session. **This stage is single-verified against Python** for the
likelihood-based quantities. Say so in the stage notes rather than
implying the usual dual standard.

---

## 5. Test matrix

### 5.1 The two reduction tests — these prove correctness

Everything else exercises shape. These two prove the implementation.

```julia
@testset "ETS(A,A,A) reproduces holt_winters at fixed parameters" begin
    y  = readdlm("fixtures/ets_y.csv")[:, 1]
    hw = holt_winters(y, 4; alpha=0.4, beta=0.1, gamma=0.3, seasonal=:additive)
    e  = fit_ets(y, 4; trend=:add, seasonal=:add,
                 fixed=(alpha=0.4, beta=0.1, gamma=0.3),
                 initialisation=:heuristic)
    @test isapprox(e.fitted, hw.fitted; atol=1e-10)
    @test isapprox(sum(e.resid.^2), hw.sse; atol=1e-8)
end

@testset "phi = 1 collapses damped onto undamped" begin
    y = readdlm("fixtures/ets_y.csv")[:, 1]
    a = fit_ets(y; trend=:damped, phi=1.0, fixed=(alpha=0.3, beta=0.1))
    b = fit_ets(y; trend=:add,              fixed=(alpha=0.3, beta=0.1))
    @test isapprox(a.fitted, b.fitted; atol=1e-10)
    @test isapprox(a.loglik, b.loglik; atol=1e-8)
end
```

`holt_winters` is already verified against R to 1e-13, so if the first
test fails the new code is the suspect.

### 5.2 Cross-language, per model

```julia
const ETS_REF = Dict(
  "ETS(A,N,N)"  => (loglik=-407.564164, aicc=821.33522, sse=6262.60017, k=2),
  "ETS(A,A,N)"  => (loglik=-394.328651, aicc=799.18362, sse=5022.88055, k=4),
  "ETS(A,N,A)"  => (loglik=-268.608601, aicc=554.51450, sse= 617.96059, k=7),
  "ETS(A,A,A)"  => (loglik=-245.248153, aicc=512.51465, sse= 418.67036, k=9),
  "ETS(A,Ad,N)" => (loglik=-394.588185, aicc=801.91973, sse=5044.65441, k=5),
  "ETS(A,Ad,A)" => (loglik=-245.483300, aicc=515.41104, sse= 420.31440, k=10))

@testset "cross-language: all six against statsmodels" begin
    y = readdlm("fixtures/ets_y.csv")[:, 1]
    for (name, spec) in SPECS
        m = fit_ets(y, 4; spec...)
        r = ETS_REF[name]
        @test isapprox(m.sse,    r.sse;    rtol=1e-4)
        @test isapprox(m.loglik, r.loglik; atol=1e-2)
        @test m.nparams == r.k
    end
end
```

**Tolerances are deliberate.** SSE is the directly comparable quantity
— match it tightly. The log-likelihood depends on the concentration
constant, which may differ; `atol=1e-2` catches a wrong formula while
tolerating a different additive constant. **If loglik differs by a
constant across all six models, that is the concentration convention
and is not a bug** — document the offset and move on.

### 5.3 Selection

```julia
@testset "AICc selects the true generating process" begin
    y = readdlm("fixtures/ets_y.csv")[:, 1]      # simulated from ETS(A,A,A)
    m = auto_ets(y, 4)
    @test m.trend == :add && m.seasonal == :add && m.damped == false
end

@testset "AICc ordering matches the reference" begin
    # the reference ranks: A,A,A < A,Ad,A < A,N,A < A,A,N < A,Ad,N < A,N,N
    y = readdlm("fixtures/ets_y.csv")[:, 1]
    fits = [fit_ets(y, 4; s...) for s in SPECS]
    @test argmin([f.aicc for f in fits]) == findfirst(==("ETS(A,A,A)"), SPEC_NAMES)
end
```

### 5.4 Parameter recovery

```julia
@testset "recovers the generating parameters within sampling error" begin
    y = readdlm("fixtures/ets_y.csv")[:, 1]   # true alpha .40 beta .10 gamma .30
    m = fit_ets(y, 4; trend=:add, seasonal=:add)
    @test isapprox(m.alpha, 0.414046; atol=0.02)
    @test isapprox(m.beta,  0.075884; atol=0.02)
    @test isapprox(m.gamma, 0.363091; atol=0.02)
end
```

Assert against the **fitted** reference, not the true values — the gap
between 0.076 and 0.10 is real sampling error and asserting on 0.10
would make the test wrong.

### 5.5 Constraint regions

```julia
@testset "traditional region rejects out-of-range parameters" begin
    y = readdlm("fixtures/ets_y.csv")[:, 1]
    @test_throws ArgumentError fit_ets(y, 4; fixed=(alpha=1.5,))
    @test_throws ArgumentError fit_ets(y, 4; fixed=(alpha=-0.1,))
end

@testset "admissible region is strictly wider than traditional" begin
    y = readdlm("fixtures/ets_y.csv")[:, 1]
    t = fit_ets(y, 4; trend=:add, seasonal=:add, constraint=:traditional)
    a = fit_ets(y, 4; trend=:add, seasonal=:add, constraint=:admissible)
    @test a.loglik >= t.loglik - 1e-8      # wider feasible set cannot fit worse
end
```

### 5.6 Forecasts and intervals

```julia
@testset "damped forecast converges, undamped does not" begin
    y = readdlm("fixtures/ets_y.csv")[:, 1]
    d = forecast(fit_ets(y; trend=:damped), 200)
    u = forecast(fit_ets(y; trend=:add),    200)
    @test abs(d.point[200] - d.point[199]) < abs(u.point[200] - u.point[199])
end

@testset "intervals widen with horizon and nest by level" begin
    f = forecast(fit_ets(readdlm("fixtures/ets_y.csv")[:,1], 4;
                         trend=:add, seasonal=:add), 12; levels=[80.0, 95.0])
    @test issorted(f.se)
    @test all(f.lower[:,2] .<= f.lower[:,1])   # 95% outside 80%
    @test all(f.upper[:,2] .>= f.upper[:,1])
end

@testset "seasonal forecast repeats with period m" begin
    f = forecast(fit_ets(readdlm("fixtures/ets_y.csv")[:,1], 4;
                         seasonal=:add), 8)
    d = diff(f.point)
    @test isapprox(d[1], d[5]; atol=1e-6)      # same seasonal increment one cycle on
end
```

### 5.7 Edge cases

```julia
@testset "edge cases" begin
    @test_throws ArgumentError fit_ets(Float64[], 4)
    @test_throws ArgumentError fit_ets(randn(3), 4; seasonal=:add)   # < 2 full cycles
    @test_throws ArgumentError fit_ets(randn(50), 4; seasonal=:mul)  # out of scope
    @test_throws ArgumentError fit_ets(randn(50), 4; error=:mul)     # out of scope
    @test_throws ArgumentError fit_ets(randn(50), 1; seasonal=:add)  # period 1
    m = fit_ets(fill(5.0, 60), 4)                                     # constant series
    @test all(isfinite, m.fitted)
end
```

**Both out-of-scope cases must throw with a message naming
multiplicative ETS as not yet implemented** — the honest-refusal
pattern this package already follows in five other places (PP's `ρ`,
KPSS's clipped p-values, Durbin-Watson's missing p-value, EGARCH's
analytic forecast, X-11's three-year minimum).

### 5.8 StatsAPI contract

`ExponentialSmoothingModel` currently has **no StatsAPI methods at
all**. Do not repeat that.

```julia
@testset "ETSModel honours the StatsAPI contract" begin
    m = fit_ets(readdlm("fixtures/ets_y.csv")[:,1], 4; trend=:add, seasonal=:add)
    @test length(coef(m)) == m.nparams
    @test size(vcov(m)) == (m.nparams, m.nparams)
    @test length(residuals(m)) == nobs(m)
    @test isfinite(loglikelihood(m)) && isfinite(aic(m)) && isfinite(bic(m))
    @test predict(m, 4).point == forecast(m, 4).point      # the alias, per 9B
end
```

---

## 6. Performance

The ETS recursion is **sequential** — step `t` needs step `t−1`, so it
cannot be vectorised across time. All the gain is in making each step
allocation-free, and in parallelising across *models* rather than
across time.

**The inner recursion.** Target zero allocations per call.

```julia
function _ets_recursion!(fitted, resid, state, y, α, β, γ, φ, m)
    @inbounds for t in eachindex(y)
        # read state, write fitted[t] and resid[t], update state in place
    end
    return nothing
end
```

- Preallocate `fitted`, `resid` and the state buffer **once** outside
  the optimiser loop and reuse across every likelihood evaluation. The
  optimiser calls this hundreds of times; allocating inside it is the
  single biggest avoidable cost.
- `@inbounds` on the main loop once indices are provably in range.
- Keep the seasonal state a plain `Vector{Float64}` of length `m` with
  a rotating index, **not** a circular-buffer type and not `circshift`
  — `circshift` allocates on every step.
- For the state vector proper (level, trend), `m ≤ 2` elements — a
  tuple or `SVector` avoids heap traffic entirely.

**Type stability is the thing to check first.** Run `@code_warntype` on
the recursion and the objective. A `Union{Nothing, Symbol}` trend
argument flowing into the hot loop will poison inference; resolve the
model form into concrete fields or type parameters **before** entering
the recursion, not inside it.

**`auto_ets` is embarrassingly parallel** — six independent fits, no
shared state. Thread over the candidate grid:

```julia
Threads.@threads for i in eachindex(candidates)
    results[i] = fit_ets(y, m; candidates[i]...)
end
```

Guard it the same way other threaded paths in this package are guarded,
and **test that the parallel and serial paths select the identical
model** — not merely an equally good one.

**Benchmark targets.** Add to the existing suite:

```julia
@benchmark fit_ets($y, 4; trend=:add, seasonal=:add)
@benchmark auto_ets($y, 4)
```

Record allocations, not just time. The recursion should show
**0 allocations**; a non-zero count means something is boxing or a
slice is copying. `@views` on any slice taken inside the loop.

A reasonable acceptance bar: `fit_ets` on this 120-point fixture should
be comfortably under a millisecond, and the recursion itself should
allocate nothing.

---

## 7. A structural decision to make first

**Whether ETS extends `ExponentialSmoothingModel` or gets its own
type.** Sharing means `holt_winters` results carry empty likelihood
fields; separating means two types for overlapping recursions.

**Recommend two types.** `holt_winters` deliberately tracks R's
`HoltWinters()`; ETS deliberately tracks R's `ets()`. They are two
different functions in R with different conventions, and collapsing
them would misrepresent both.

Decide before writing code — it determines the API shape.

---

## 8. Checklist

- [ ] Type decision made and recorded
- [ ] Six linear models fittable
- [ ] Both damped recursions, φ and its constraint region
- [ ] Concentrated likelihood; `aic`/`aicc`/`bic`
- [ ] `:traditional` and `:admissible` both available
- [ ] `auto_ets` selecting by AICc
- [ ] Analytic prediction intervals
- [ ] **Both reduction tests pass** (5.1)
- [ ] All six match the reference SSE to `rtol=1e-4` (5.2)
- [ ] AICc selects ETS(A,A,A) on the fixture (5.3)
- [ ] Multiplicative error and seasonal throw, naming the limitation
- [ ] Full StatsAPI contract including `vcov`; `forecast` aliases `predict`
- [ ] `show` prints `ETS(A,Ad,A)` notation
- [ ] **Recursion allocates zero**; `@code_warntype` clean
- [ ] Parallel and serial `auto_ets` select the identical model
- [ ] Stage notes state the single-verified status
- [ ] British-Indian spelling in docstrings

---

## 9. What to do

1. Make the type decision in section 7.
2. **Verify the concentrated-likelihood constant against `ets()`'s
   source before building AIC on it.** If it differs, selection still
   works and comparison against R silently does not.
3. Implement in a new `src/ets.jl`; do **not** modify
   `holtwinters.jl` — the equivalence test depends on it staying
   independent.
4. Run the two reduction tests first.
5. Profile before optimising, then check allocations.
6. Update `development-sequence.md`: record that the diffuse-init
   dependency was never real, and that multiplicative ETS is
   deliberately deferred rather than overlooked.
