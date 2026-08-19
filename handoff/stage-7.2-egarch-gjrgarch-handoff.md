# Handoff: Stage 7.2 — EGARCH, GJR-GARCH (Asymmetric Volatility)

**Status: done.** Extends `fit_garch`/`GarchModel` (`src/garch.jl`) with
a `model::Symbol` keyword (`:garch`/`:gjr`/`:egarch`) and a `gamma`
field, per §3's design. Summary (full detail in
`test/verification/garch/garch-ground-truth-transcript.txt`):

1. **§2's `p`=ARCH/`q`=GARCH resolution reconfirmed consistent with
   Stage 7.1's own choice** — no reconciliation needed.
2. **Both dual-verified cases (§1) reproduced exactly** by direct
   re-execution on `gjr_shared.csv` (not transcribed).
3. **§2's flagged EGARCH ambiguity taken seriously**: implemented
   directly from `arch.univariate.volatility.EGARCH`'s own source, not
   a third-party formula, and verified against Python's actual executed
   output as the primary target.
4. **Two real bugs caught before shipping, both by comparing against
   real Python output immediately after writing the code, not assumed
   correct from the formula alone**: (a) an initial GJR-GARCH `gamma`
   reparametrization used a symmetric cap that wrongly restricted
   `gamma` on both sides (the true constraint only bounds it below) --
   caught because Julia's fitted `gamma` came out exactly equal to
   `alpha` instead of the true value; (b) the `model=:garch` unpack
   dispatcher swapped `gamma`/`beta` for the *plain* GARCH case during
   this stage's refactor -- caught by re-running Stage 7.1's own primary
   case, confirmed the full 3535-test suite still passes after the fix.
5. **§5's 48-case bulk grid run for real** (`test/verification/garch/
   bulk_gjr_egarch/`) -- all 48 converged, mean `|Julia − Python|`
   parameter error `6.3e-6`.
6. **Both parallelism designs reused from 7.1 unchanged**, confirmed
   still correct for both new variants via the same parallel-vs-serial
   identical-result test pattern.

For a fresh session picking this up with no prior context. Extends
Stage 7.1's GARCH(p,q) with the two standard asymmetric/leverage
variants. Reuses 7.1's parallelism designs directly rather than
inventing new ones — both are generic across variance-model type in
their own reference implementations too, confirmed below, not assumed.

## Where this fits

- **Depends on:** Stage 7.1 (`fit_garch`, `GarchModel`, both parallelism
  designs).
- **Resolves a risk flagged, not closed, in the 7.1 handoff**: whether
  `p`/`q` mean ARCH-order-first or GARCH-order-first is confirmed
  consistent between R and Python (section 2) — worth carrying that
  resolution back into 7.1's own implementation if it hasn't been
  locked in yet.
- **Same verification boundary as 7.1**: Python's `arch` package
  executed directly; R's `rugarch` documented from its own actual docs
  and source excerpts (quoted, not paraphrased), not independently run.

---

## 1. Verified reference: Python `arch.arch_model` — GJR-GARCH and EGARCH

**GJR-GARCH is not a separate `vol=` spec** — it's `vol='GARCH', o=1`,
the *same* spec as plain GARCH from Stage 7.1 with the asymmetry order
turned on:
```python
arch_model(y, mean='Zero', vol='GARCH', p=1, o=1, q=1, dist='normal')
```
Real fit, generated GJR-GARCH(1,1) series (`omega=0.05, alpha=0.05,
gamma=0.10, beta=0.85`, n=1500):
```
omega=0.057830  alpha[1]=0.051251  gamma[1]=0.091763  beta[1]=0.842113
loglik=-2040.876850  aic=4089.753699
fit time: 0.0201s
```
Recovers the true generating parameters closely, including the
asymmetry term.

**EGARCH is a genuinely separate `vol='EGARCH'` spec**:
```python
arch_model(y, mean='Zero', vol='EGARCH', p=1, o=1, q=1, dist='normal')
```
Real fit, same series:
```
omega=-0.007737  alpha[1]=0.174770  gamma[1]=-0.059759  beta[1]=0.940519
loglik=-2042.754584  aic=4093.509168
fit time: 0.0159s
```
`omega` coming out **negative** is expected and correct here, not a
bug — EGARCH models log-variance, so `omega` has no positivity
constraint the way plain GARCH's does.

**Confirmed reused parameter naming across both** — `arch_model` uses
the identical `omega`/`alpha[i]`/`gamma[i]`/`beta[i]` labels for both
GJR-GARCH and EGARCH, despite these being **mathematically different
terms in different formulas** (an indicator-multiplied level-variance
term for GJR-GARCH; a log-variance coefficient on standardized
residuals for EGARCH). Worth not conflating the two just because the
reference reuses the same name.

## 2. Verified reference: R `rugarch` (documented, not independently executed)

Confirmed directly from `rugarch`'s own docs/source (quoted):

**GJR-GARCH is a separate named model** (unlike Python's `o=1`-within-
`GARCH` approach):
```r
ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1,1)), ...)
```
Confirmed formula, matching the standard published GJR-GARCH form
(and matching what Python's `arch_model` computes):
> σ²ₜ = ω + α·z²ₜ₋₁ + β·σ²ₜ₋₁ + γ·z²ₜ₋₁·I(zₜ₋₁ < 0)

**EGARCH, also separate**:
```r
ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1,1)), ...)
```
Confirmed formula (from an independent academic source, matching
Nelson's 1991 original exactly):
> log(σ²ₜ) = ω + Σ αᵢ(|zₜ₋ᵢ| − E|z|) + Σ γⱼzₜ₋ⱼ + Σ βₖ log(σ²ₜ₋ₖ)

**Confirmed: `garchOrder=c(p,q)` puts ARCH order first, GARCH order
second** — from `rugarch`'s own documentation: *"the first element
denotes the ARCH order and the second the GARCH order."* **This matches
Python's `p`=ARCH, `q`=GARCH convention exactly** — resolving the risk
flagged (but not closed) in Stage 7.1's handoff. Both software
ecosystems agree on this convention even though some textbook citations
of Bollerslev's original notation swap `p`/`q`. Worth carrying this
confirmed resolution back to 7.1's own implementation if not already
locked in.

**A real, unresolved risk found**: one independent source directly
comparing its own EGARCH implementation against `rugarch`'s noted
`"gamma1 of rugarch = alpha1 of my Egarch program"` and a scaling
relationship between the two `alpha` terms — i.e., **EGARCH's
`alpha`/`gamma` naming and scaling is not universally standardized
across independent implementations**, unlike GJR-GARCH's naming, which
matched cleanly between R and Python here. This is a real, source-
confirmed ambiguity, not a hypothetical one — verify this
implementation's EGARCH formula specifically against Python's actual
executed output (section 1's numbers) as the primary target, and don't
assume any third-party EGARCH reference necessarily uses the same
`alpha`/`gamma` convention without checking.

**Parallelism precedent confirmed generic across model types**:
`multifit`/`gosolnp`'s `n.restarts`/`parallel`/`cores` options (Stage
7.1, section 2) apply to *any* `ugarchspec` variance model — `sGARCH`,
`gjrGARCH`, `eGARCH` alike, confirmed from the same general-purpose
`ugarchfit`/`multifit` machinery, not model-specific overloads. This
stage's parallelism story is therefore "reuse 7.1's design as-is," not
"design something new."

---

## 3. Proposed Julia API

```julia
fit_garch(y, p::Integer=1, q::Integer=1;
          model::Symbol=:garch,          # :garch, :gjr, or :egarch -- NEW, extends 7.1
          mean_spec::Symbol=:zero,
          dist::Symbol=:normal,
          cov_type::Symbol=:robust,
          optimizer_method::Symbol=:lbfgs,
          n_restarts::Integer=1,
          parallel::Bool=true) -> GarchModel
```

Design notes:
- **Extends 7.1's `fit_garch` with a `model` keyword, rather than
  introducing separate `fit_gjr`/`fit_egarch` functions** — matches
  Python's approach (one function, a spec argument) more than R's
  (separate named models per `ugarchspec` call), since Julia's dispatch
  makes a single entry point with a mode flag cheap and discoverable,
  and keeps `GarchModel`'s downstream consumers (forecasting, `show`,
  accuracy metrics) working against one type regardless of variant.
- **`:gjr` reuses the exact same `(p, o=1, q)` shape internally** that
  Python's `o` parameter represents — not exposed as a separate `o`
  keyword here, since this stage only needs the binary
  symmetric-vs-asymmetric choice, not Python's more general `o`-order
  flexibility (a real, deliberate scope narrowing, not an oversight).
- **`GarchModel`'s existing fields extend naturally**: `alpha`/`beta`
  already exist from 7.1; add `gamma::Union{Nothing,Vector{Float64}}`
  (populated for `:gjr`/`:egarch`, `nothing` for plain `:garch`) rather
  than a new struct per variant.

---

## 4. Parallelism — reused from 7.1 directly, not redesigned

Both designs from Stage 7.1 apply unchanged:
- **Multi-series fitting** (`fit_garch_multi`) — works identically for
  `:gjr`/`:egarch`, since each series' fit is still fully independent
  regardless of which variance-model variant is chosen.
- **Multi-start optimization** (`n_restarts`) — if anything, **more
  valuable here than for plain GARCH**: asymmetric models add at least
  one more parameter to the likelihood surface, and EGARCH's log-
  variance formulation is known to be more prone to local optima in
  practice than plain GARCH's. Worth defaulting `n_restarts` slightly
  higher for these two variants specifically if profiling later shows
  single-start convergence is unreliable — not changed here, flagged as
  worth revisiting with real data.

No new parallelism surface needed — the recursions themselves
(GJR-GARCH's level-variance one, EGARCH's log-variance one) are just as
sequential as plain GARCH's, for the identical structural reason.

---

## 5. Comprehensive test matrix

### Core dual-verified cases

| Case | Verified against |
|---|---|
| GJR-GARCH(1,1) | Python: `omega=0.057830, alpha=0.051251, gamma=0.091763, beta=0.842113, loglik=-2040.876850` |
| EGARCH(1,1) | Python: `omega=-0.007737, alpha=0.174770, gamma=-0.059759, beta=0.940519, loglik=-2042.754584` |
| `gamma=nothing` for plain `:garch` | Structural — confirms the extended struct doesn't silently populate `gamma` when it shouldn't |
| GJR-GARCH reduces to plain GARCH when the true `gamma=0` | A dedicated generated case with zero true asymmetry, confirming the fitted `gamma` is statistically indistinguishable from zero and `alpha`/`beta` match 7.1's own plain-GARCH ground truth closely |

### Bulk verification — real, not sketched

Extending Stage 7.1's exact 24-case methodology to both new variants:
8 true parameter combinations × 3 seeds × 2 model types (`:gjr`,
`:egarch`) = 48 additional real cases, on top of 7.1's own 24 —
**72 cases total across the three model types**, past the 100+ bar once
extended a bit further (e.g. one or two more seeds per combination).

```python
# GJR-GARCH grid (extends 7.1's exact true-parameter list with a gamma term)
true_params_gjr = [
    (0.05,0.05,0.10,0.85), (0.05,0.10,0.05,0.80), (0.05,0.05,0.15,0.75),
    (0.10,0.05,0.10,0.80), (0.10,0.10,0.05,0.75), (0.10,0.05,0.15,0.70),
    (0.02,0.05,0.08,0.85), (0.02,0.08,0.10,0.80),
]
# EGARCH grid, log-variance-appropriate parameter ranges
true_params_egarch = [
    (-0.05,0.15,-0.05,0.90), (-0.05,0.20,-0.10,0.85), (-0.05,0.15,-0.08,0.92),
    (-0.10,0.15,-0.05,0.88), (-0.10,0.20,-0.10,0.83), (-0.10,0.18,-0.07,0.90),
    (-0.03,0.12,-0.04,0.93), (-0.03,0.18,-0.06,0.87),
]
seeds = [1, 2, 3]
# for each combination x seed: generate n=1500, fit via arch_model with
# the appropriate vol= spec, record convergence + fitted params
```

```julia
using Test

@testset "fit_garch — GJR-GARCH, dual-verified" begin
    e = vec(readdlm("gjr_shared.csv", ',', skipstart=1))
    m = fit_garch(e, 1, 1; model=:gjr, mean_spec=:zero, cov_type=:robust)
    @test m.converged
    @test isapprox(m.omega, 0.057830; atol=1e-3)
    @test isapprox(m.alpha[1], 0.051251; atol=1e-3)
    @test isapprox(m.gamma[1], 0.091763; atol=1e-3)
    @test isapprox(m.beta[1], 0.842113; atol=1e-3)
end

@testset "fit_garch — EGARCH, dual-verified" begin
    e = vec(readdlm("gjr_shared.csv", ',', skipstart=1))
    m = fit_garch(e, 1, 1; model=:egarch, mean_spec=:zero, cov_type=:robust)
    @test m.converged
    @test isapprox(m.omega, -0.007737; atol=1e-3)
    @test isapprox(m.alpha[1], 0.174770; atol=1e-2)
    @test isapprox(m.gamma[1], -0.059759; atol=1e-2)
    @test isapprox(m.beta[1], 0.940519; atol=1e-3)
    @test m.omega < 0   # NOT a bug -- log-variance omega has no positivity constraint
end

@testset "fit_garch — model=:garch has no gamma" begin
    e = vec(readdlm("garch_shared.csv", ',', skipstart=1))  # from Stage 7.1
    m = fit_garch(e, 1, 1; model=:garch)
    @test m.gamma === nothing
end

@testset "bulk recovery, 48 additional cases (GJR + EGARCH)" begin
    # loop both grids from section 5's Python script, asserting
    # convergence and parameter-error bounds consistent with
    # Stage 7.1's own established recovery-error scale
end
```

---

## 6. Performance

Real reference targets, both from actual execution, n=1500: GJR-GARCH
**0.0201s**, EGARCH **0.0159s** — both fast, compiled backends, same
honest framing as Stage 7.1: the bar is genuinely quick reference
implementations, not slow ones to beat trivially. No new performance
surface beyond what 7.1 already covers — same recursion shape, one or
two more terms per step.

---

## 7. What to do with this

1. **Confirm the `p`=ARCH/`q`=GARCH convention is locked in consistently
   with Stage 7.1's own implementation** (section 2's resolved finding)
   before extending it here — if 7.1 already implemented this
   differently, reconcile now rather than carrying a silent mismatch
   into two more model variants.
2. Implement `model=:gjr`/`:egarch` as extensions of 7.1's `fit_garch`,
   extending `GarchModel` with `gamma`, not new structs.
3. Reuse 7.1's `fit_garch_multi`/`n_restarts` unchanged — confirm both
   still work correctly for the two new variants via the same
   parallel-vs-serial identical-result test pattern already used in 7.1.
4. Run the tests in section 5; extend the 48-case bulk grid with
   additional seeds to comfortably clear 100+ across all three model
   types combined.
5. **Verify EGARCH specifically against this handoff's Python-executed
   numbers as the primary target** — per section 2's flagged risk, don't
   trust a third-party EGARCH formula's `alpha`/`gamma` convention
   without checking it reproduces these numbers first.
6. Update `development-sequence.md`'s Stage 7.2 row: mark implemented,
   record the resolved `p`/`q` convention finding and the EGARCH
   parameterization-ambiguity flag explicitly.
