# Handoff: Stage 7.3 — Volatility Forecasting

**Status: done.** Implemented as `forecast_volatility`/
`VolatilityForecast` (`src/garchforecast.jl`), per §3's design. Summary
(full detail in
`test/verification/garch/garch-ground-truth-transcript.txt`):

1. **§1's EGARCH-horizon>1-throws finding reconfirmed** by direct
   execution, exact error message matched.
2. **A real transcription typo caught in this handoff's own §1/§5**:
   the GJR-GARCH h=5 analytic forecast's last transcribed value
   (`1.7777`) is a digit-transposition of the true value (`1.6777`),
   confirmed by re-executing real `arch` directly, twice. Used the
   corrected value as the actual test target, not the transcribed one.
3. **§3's `method=:auto` implemented as designed**, resolving to
   `:analytic` for `:garch`/`:gjr` and `:simulation` for `:egarch`.
4. **§1's analytic recursion implemented directly from `arch`'s own
   `GARCH._analytic_forecast` source**, generalized for real multi-lag
   `p`/`q`, not just the `(1,1)` case the handoff's own informal sketch
   described.
5. **§5's convergence check run for real**: Julia's own simulation vs.
   its own analytic forecast, `50000` paths, max abs diff `0.000725`
   (handoff's own reference figure for real `arch`: `0.00098`).
6. **§4's parallelism implemented exactly as designed and precedent-
   confirmed** (`rugarch`'s `ugarchboot` docs actively recommend
   parallelizing this specific workload).
7. **§5's 216-check bulk grid run for real**, reusing all 72
   already-fitted models from Stage 7.1/7.2's own bulk sweeps rather
   than generating new data — all pass.

For a fresh session picking this up with no prior context. This stage
has a genuinely different parallelism story from 7.1/7.2 — not
multi-series fitting, but **thousands of independent Monte Carlo
simulation paths**, needed because EGARCH has no closed-form multi-step
forecast at all. Confirmed by direct execution (a real error message,
not a guess) and by explicit, stated precedent in R's own bootstrap
forecasting docs.

## Where this fits

- **Depends on:** Stage 7.1 (`GarchModel`), Stage 7.2 (`model` variants).
- **Verification boundary**: same as 7.1/7.2 — Python's `arch` executed
  directly; R's `rugarch` from its own documentation and source
  (quoted), not independently run.

---

## 1. Verified reference: Python `ARCHModelResult.forecast`

```python
forecast(params=None, horizon=1, start=None, align='origin',
         method='analytic', simulations=1000, rng=None,
         random_state=None, *, reindex=False, x=None)
```

Confirmed via `inspect.signature`. `method`: `'analytic'` (default),
`'simulation'`, or `'bootstrap'`.

**Real, execution-confirmed finding — not assumed**: `method='analytic'`
works fine for GARCH and GJR-GARCH at any horizon, but **throws a clear
error for EGARCH beyond one step**:
```
GARCH,     horizon=10, analytic: [0.7852, 0.7957, 0.8058, ..., 0.8657]  -- works
GJR-GARCH, horizon=5,  analytic: [1.8846, 1.8279, 1.7747, 1.7247, 1.7777]  -- works
EGARCH,    horizon=5,  analytic: ValueError: "Analytic forecasts not
                                   available for horizon > 1"
```
This is the real reason simulation-based forecasting isn't an
optional extra for this stage — it's the **only** way to get a
multi-step EGARCH forecast at all, since its log-variance recursion has
no closed-form multi-step expectation the way GARCH's and GJR-GARCH's
(under a symmetric-innovation assumption) do.

**Simulation timing scales linearly with path count** — confirmed
directly:
```
EGARCH, horizon=5, simulations=1000:  0.0018s
EGARCH, horizon=5, simulations=10000: 0.0181s   (~10x paths, ~10x time)
```
Exactly the structure of an embarrassingly parallel workload — cost is
almost entirely per-path, paths share no state.

**Simulation converges to the analytic formula where both exist** — a
genuine correctness check on the simulation machinery itself, not just
a nice property: GARCH's analytic 5-step forecast vs. its simulation-
based forecast at `simulations=50000` agree to within Monte Carlo noise
(max absolute difference `0.00098`). This is the test to run once this
stage's simulation engine exists, before trusting it on EGARCH where
there's no analytic answer to check against.

## 2. Documented reference: R `rugarch` (not independently executed)

`ugarchforecast(fitORspec, n.ahead=10, n.roll=0, ...)` — the general
forecast entry point. For models needing simulation (confirmed directly
from `rugarch`'s realized-GARCH documentation, generalizes to EGARCH's
same structural need): *"n.ahead>1 forecasts contain uncertainty...
simulation methods are used... `n.sim` for the number of random samples
to use per period."*

**Direct, explicit precedent for parallelizing exactly this** — from
`ugarchboot`'s own documentation, quoted: *"This process, while more
accurate, is very time consuming which is why choice of parallel
computation via a cluster... is available and recommended."* Real
example default: `n.bootpred = 2000` simulation paths. This is stronger
precedent than 7.1/7.2's `multifit`/`gosolnp` findings — R's own docs
don't just support parallelizing this, they actively recommend it,
unprompted, specifically because it's slow otherwise.

---

## 3. Proposed Julia API

```julia
forecast_volatility(m::GarchModel, horizon::Integer=1;
                     method::Symbol=:auto,       # :auto, :analytic, or :simulation
                     simulations::Integer=1000,  # matches Python's default exactly
                     parallel::Bool=true) -> VolatilityForecast
```

Design notes:
- **`method=:auto` (new, not in either reference)**: picks `:analytic`
  when available (`model in (:garch, :gjr)`, matching the confirmed
  working cases from section 1) and `:simulation` automatically when
  it isn't (`model == :egarch`) — rather than making every EGARCH caller
  remember to pass `method=:simulation` explicitly or hit a runtime
  error. `:analytic` explicitly requested on an EGARCH model should
  still throw a clear, named error (matching Python's own behavior,
  not silently substituting simulation under the requested name) —
  same "don't silently change what was asked for" discipline as every
  prior stage's honest-gap handling.
- **`simulations::Integer=1000`**: matches Python's exact default
  value, not just the concept — a real, checkable number, not a
  round-sounding guess.
- **`parallel::Bool=true`**: see section 4.

### `VolatilityForecast`

```julia
struct VolatilityForecast
    variance::Vector{Float64}        # point forecast, one per horizon step
    variance_paths::Union{Nothing,Matrix{Float64}}  # simulations x horizon, only when method=:simulation
    method::Symbol
    simulations::Union{Nothing,Int}
    horizon::Int
end
```
`variance_paths` being populated (simulation) vs. `nothing` (analytic)
directly reflects whether the point forecast came with a real
distribution of simulated outcomes behind it — useful for anyone
wanting prediction intervals on the variance forecast itself, not just
the point estimate.

---

## 4. Parallelism — the real target of this stage, precedent-confirmed twice over

**What's sequential**: each individual simulated path's own within-path
recursion (`h_t` depends on `h_{t-1}` along that one path) — same
structural reason as every fitting recursion in this project.

**What's embarrassingly parallel, and is this stage's actual point**:
the `simulations` independent paths themselves share no state with each
other at all. `Threads.@threads` over the simulation index, guarded the
same way as every other parallel design in this project
(`Threads.nthreads() > 1`, and a minimum path count — simulation counts
in the hundreds-to-thousands make this an easy bar to clear, unlike the
small-series-count guards needed for 7.1's multi-series design).

```julia
function _simulate_paths(m::GarchModel, horizon::Integer, simulations::Integer; parallel::Bool=true)
    paths = Matrix{Float64}(undef, simulations, horizon)
    use_threads = parallel && Threads.nthreads() > 1 && simulations >= 100
    if use_threads
        Threads.@threads for s in 1:simulations
            paths[s, :] = _simulate_one_path(m, horizon)
        end
    else
        for s in 1:simulations
            paths[s, :] = _simulate_one_path(m, horizon)
        end
    end
    return paths
end
```

This is the strongest parallelism case in the whole GARCH module so
far — both because the workload is naturally large (thousands of
independent paths, not a handful of series) and because it's the one
place a reference implementation's own documentation explicitly
recommends parallelizing it rather than this project inferring the
opportunity independently.

---

## 5. Comprehensive test matrix

### Core dual-verified/correctness cases

| Case | Verified against |
|---|---|
| GARCH analytic, horizon=10 | Python: `[0.7852, 0.7957, ..., 0.8657]` |
| GJR-GARCH analytic, horizon=5 | Python: `[1.8846, 1.8279, 1.7747, 1.7247, 1.7777]` |
| EGARCH analytic, horizon>1 | Must throw, matching Python's confirmed `ValueError` |
| EGARCH simulation, horizon=5 | Python: `[1.6389, 1.5938, 1.5509, 1.5166, 1.4584]` at `simulations=1000` |
| GARCH simulation vs. analytic agreement | Max abs diff `0.00098` at `simulations=50000` -- the correctness check on the simulation engine itself |
| `method=:auto` picks correctly | `:garch`/`:gjr` -> analytic path taken; `:egarch` -> simulation path taken, without the caller specifying either |
| Parallel simulation matches serial | Same seeded RNG stream, `parallel=true` vs `parallel=false` give statistically indistinguishable (not necessarily bit-identical, given thread-scheduling-dependent RNG draw order) point forecasts |

### Bulk verification — real, extending 7.1/7.2's already-executed grids

Reuse the 24 GARCH cases (7.1) and 48 GJR/EGARCH cases (7.2) — for
each already-fitted model in those grids, forecast `horizon in {1, 5,
10}` and confirm: (a) analytic and simulation agree for GARCH/GJR at
each horizon, (b) EGARCH's simulation-based forecast is internally
consistent (variance stays positive, converges toward a stable level
rather than diverging) across all cases. `24 + 48 = 72` models x 3
horizons = **216 forecast checks**, comfortably past the 100+ bar,
built entirely on data already generated and verified in 7.1/7.2 rather
than a new grid.

```julia
using Test

@testset "forecast_volatility — core cases" begin
    e = vec(readdlm("garch_shared.csv", ',', skipstart=1))
    m = fit_garch(e, 1, 1)  # from Stage 7.1

    fc_a = forecast_volatility(m, 10; method=:analytic)
    @test isapprox(fc_a.variance, [0.7852,0.7957,0.8058,0.8155,0.8248,
                                     0.8337,0.8422,0.8503,0.8582,0.8657]; atol=1e-3)

    fc_s = forecast_volatility(m, 5; method=:simulation, simulations=50000)
    fc_a5 = forecast_volatility(m, 5; method=:analytic)
    @test maximum(abs.(fc_s.variance .- fc_a5.variance)) < 0.005  # Monte Carlo tolerance

    e_ego = vec(readdlm("gjr_shared.csv", ',', skipstart=1))
    m_ego = fit_garch(e_ego, 1, 1; model=:egarch)  # from Stage 7.2
    @test_throws Exception forecast_volatility(m_ego, 5; method=:analytic)

    fc_ego = forecast_volatility(m_ego, 5; method=:auto)
    @test fc_ego.method == :simulation   # auto correctly picked simulation
    @test all(fc_ego.variance .> 0)

    fc_par = forecast_volatility(m_ego, 5; simulations=5000, parallel=true)
    fc_serial = forecast_volatility(m_ego, 5; simulations=5000, parallel=false)
    @test isapprox(mean(fc_par.variance), mean(fc_serial.variance); atol=0.05)
end

@testset "bulk: 72 models x 3 horizons" begin
    # loop the 7.1/7.2 grids; for each fitted model, forecast at
    # horizon in (1,5,10); assert positivity throughout and, for
    # garch/gjr, analytic-vs-simulation agreement per the tolerance above
end
```

---

## 6. Performance

Real reference targets, both from direct execution: EGARCH 5-step
simulation forecast, **0.0018s at 1000 paths, 0.0181s at 10000 paths**
— genuinely fast even serially, meaning the parallel design's real
value shows up at the larger path counts serious risk-management use
tends to want (10,000+ paths for tail-risk/VaR-style applications), not
necessarily at the small default. Profile the actual crossover point
where `parallel=true` measurably helps once this is running, rather
than assuming it matters at every path count — same "don't
pre-optimize, verify with a profile" discipline as every prior stage.

---

## 7. What to do with this

1. Implement `forecast_volatility`/`VolatilityForecast` and the
   analytic recursions for `:garch`/`:gjr` (straightforward, closed-form
   -- see any standard GARCH forecasting reference for the exact
   `omega + (alpha+beta)*h` / `omega + (alpha+0.5*gamma+beta)*h`
   multi-step recursions, the `0.5` from symmetric-innovation
   probability).
2. Implement the simulation engine (`_simulate_paths`/
   `_simulate_one_path`), then **immediately run the GARCH
   simulation-vs-analytic convergence check (section 5)** before trusting
   it on EGARCH, where there's nothing to check it against directly.
3. Implement `method=:auto` and the explicit-analytic-on-EGARCH error.
4. Implement the parallel path simulation per section 4.
5. Run the bulk 216-check suite reusing 7.1/7.2's fitted models.
6. Update `development-sequence.md`'s Stage 7.3 row: mark implemented,
   record the EGARCH-analytic-impossibility finding and both precedent
   citations (7.1/7.2's `multifit`/`gosolnp`, this stage's stronger
   `ugarchboot`-recommends-parallel finding) explicitly.
