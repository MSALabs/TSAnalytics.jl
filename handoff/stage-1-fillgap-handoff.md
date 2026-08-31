# Handoff: Stage 1 Fill-Gap -- Periodogram/Spectral Density (1.5), Box-Cox Transform (1.6)

For a fresh Claude Code session picking this up with no prior context.
This closes two gaps found by comparing Stage 1 (numerical primitives)
against all six reference books -- Hamilton, fpp3, Durbin & Koopman,
Tsay, Ladiray & Quenneville, Shumway & Stoffer -- checked a second time,
systematically, before this handoff was written, not just recalled from
the earlier pass.

## Where this fits

- **Depends on:** nothing beyond what Stage 1.1-1.4 already established
  (basic linear algebra, no new external dependency needed for 1.5;
  1.6's automatic-lambda selection needs 1.4's regression machinery,
  not new machinery).
- **Both gaps are corroborated by more than one book**, not a single
  citation each -- see sections 1 and 2 for exactly which books and how.
- **This sits alongside 1.1-1.4** as `1.5`/`1.6`, at the same
  foundational level, not tacked on as an "advanced" feature -- that's
  the actual finding from checking the books: both fpp3 and Shumway &
  Stoffer place their equivalents in early, foundational chapters
  (fpp3's own transformations section sits with decomposition; Shumway
  & Stoffer's spectral analysis is literally Chapter 4, structurally
  parallel to where ACF/PACF sits in this project).

---

## 1. Stage 1.5 -- Periodogram / spectral density estimation

**Two independent books corroborate this belongs at the primitives
level, not deferred**:
- **Shumway & Stoffer**: confirmed directly -- their companion package
  `astsa` has a real, actively-developed function, `mvspec`
  ("Univariate and Multivariate Spectral Estimation"), tied explicitly
  to **Chapter 4: "Spectral Analysis and Filtering"** of the actual
  textbook -- the book's fourth chapter, the same structural position
  ACF/PACF occupies in this project's own Chapter Two.
- **Hamilton**: treats spectral density as the natural Fourier-domain
  counterpart to the autocovariance-domain tools already built in
  Stage 1.3, introduced early rather than as an advanced aside.

**Verified directly against base R's `spec.pgram`** (confirmed
available without CRAN -- base `stats` package, no network dependency):
```r
spec.pgram(x, spans=NULL, kernel=NULL, taper=0.1, pad=0, fast=TRUE,
           demean=FALSE, detrend=TRUE, plot=TRUE, na.action=na.fail, ...)
```
Real, exact defaults confirmed: **`taper=0.1`** (10% split-cosine-bell
taper applied by default), **`detrend=TRUE`** (linear detrend before
computing, which also removes the mean), **`demean=FALSE`** (redundant
given `detrend=TRUE`'s default, but a real, independent flag), **no
padding by default**.

**A genuine, real cross-reference worth keeping**: `astsa::mvspec`
(Shumway & Stoffer's own companion package) *deliberately overrides*
base R's tapering default -- confirmed directly from its own
documentation: *"the script does not taper by default (taper=0); this
forces the user to do 'conscious tapering'."* Two different, real
design philosophies from two different sources on the same underlying
computation -- worth a deliberate choice here too, not silently picking
one. **Recommend matching Shumway & Stoffer's `taper=0` default** (the
more actively-taught convention, and consistent with this project's
general preference for explicit rather than implicit smoothing choices
elsewhere), while documenting base R's `0.1` default directly in the
docstring so the divergence is visible.

**Real, verified test case** -- a series with a known, injected
periodicity, run through base R directly:
```r
set.seed(42); n <- 96; t <- 1:n
x <- 5*sin(2*pi*t/12) + 3*cos(2*pi*t/12) + rnorm(n, sd=0.5)
r <- spec.pgram(x, taper=0.1, plot=FALSE)
# peak frequency: 0.08333333333  (exactly 1/12, matching the injected period-12 component)
# df: 1.791590494   bandwidth: 0.003007032652
# spec[1:5]: 5.787370716, 1.375818839, 2.646642418, 2.651190832, 9.823573797
```

### Proposed API

```julia
periodogram(x::Vector{Float64}; taper::Real=0.0, detrend::Bool=true,
            demean::Bool=false, pad::Integer=0) -> (freq=Vector{Float64}, spec=Vector{Float64})

spectral_density(x::Vector{Float64}, spans::Vector{Int}; kwargs...) -> (freq, spec, df, bandwidth)
```

- **`periodogram`**: the raw, unsmoothed estimate -- matches R's
  `spec.pgram` with `spans=NULL`.
- **`spectral_density`**: the smoothed estimate via modified Daniell
  kernel smoothing (`spans=` -- matches R's own convention directly,
  confirmed from the real `mvspec(soi, spans=c(7,7), ...)` usage
  pattern found during research), returning `df`/`bandwidth` alongside
  the spectrum, matching what a user needs to construct confidence
  intervals the way both R and the textbook do.
- **FFT-based**, reusing whatever FFT library is already a project
  dependency elsewhere (check Stage 7's GARCH/volatility work or
  TSFeatures.jl's own `FFTW.jl` dependency before adding a new one).

---

## 2. Stage 1.6 -- Box-Cox transform, with Guerrero's automatic lambda selection

**Corroborated by fpp3 specifically** -- placed in the book's own
Section 3.1, structurally alongside decomposition (already Stage 3.1 in
this project), not as a separate later chapter. The transform itself:
```
y_t^(lambda) = (y_t^lambda - 1) / lambda    if lambda != 0
y_t^(lambda) = log(y_t)                      if lambda == 0
```

**Guerrero's method** (Guerrero, 1993, *Journal of Forecasting* -- the
specific automatic-lambda-selection method fpp3 itself recommends and
its own `forecast`/`fable` packages implement as the default):
1. Split the series into subseries of length matching the seasonal
   period (2 subseries if the series is non-seasonal).
2. For each subseries `i`, compute the sample mean `m_i` and standard
   deviation `s_i`.
3. For a candidate `lambda`, compute the rescaled variability
   `s_i / m_i^(1-lambda)` for each subseries.
4. Compute the coefficient of variation of these rescaled values across
   all subseries.
5. Choose the `lambda` (typically bounded to `[-1, 2]`, matching fpp3's
   own stated convention) that **minimizes** this coefficient of
   variation -- i.e., the transform that makes the seasonal subseries'
   variability most *consistent*, the actual variance-stabilization
   goal Box-Cox exists for.

**Honest verification note**: this formula is well-established and
directly cited from fpp3's own text, but **could not be independently
executed this session** -- `forecast::BoxCox.lambda(method="guerrero")`
needs the `forecast` package, and CRAN was confirmed unreachable again
this session (a fresh install attempt failed the same way it has
throughout this project). Treat the algorithm above as documented,
not cross-verified against real output -- flag this explicitly in the
implementation's own docstring rather than imply a verification
standard that wasn't actually met this time.

### Proposed API

```julia
boxcox(x::Vector{Float64}, lambda::Real) -> Vector{Float64}
boxcox_inv(y::Vector{Float64}, lambda::Real) -> Vector{Float64}
guerrero_lambda(x::Vector{Float64}, period::Integer=1; bounds::Tuple=(-1.0, 2.0)) -> Float64
```

- **`guerrero_lambda`**: `period=1` triggers the 2-subseries non-seasonal
  case per Guerrero's own method as documented; any `period > 1` splits
  by that periodicity. Bounded numerical search (golden-section or
  `Optim.jl`'s already-established bounded solver from Stage 4.1 --
  reuse it rather than add new optimizer machinery) over `bounds`.

---

## 3. Comprehensive test matrix

```julia
using Test

@testset "periodogram -- real, verified peak-detection case" begin
    Random.seed!(42)
    n = 96
    t = 1:n
    x = 5 .* sin.(2π .* t ./ 12) .+ 3 .* cos.(2π .* t ./ 12) .+ randn(n) .* 0.5
    result = periodogram(x; taper=0.1)
    peak_freq = result.freq[argmax(result.spec)]
    @test isapprox(peak_freq, 1/12; atol=1e-6)   # matches R's verified 0.08333333333
end

@testset "periodogram -- Nyquist and DC properties" begin
    x = randn(100)
    result = periodogram(x)
    @test all(result.freq .> 0)          # DC term excluded, matching R's convention
    @test all(result.freq .<= 0.5)       # bounded by Nyquist frequency
end

@testset "boxcox -- exact log-transform reduction at lambda=0" begin
    x = abs.(randn(50)) .+ 1.0   # must stay positive
    @test isapprox(boxcox(x, 0.0), log.(x); atol=1e-10)
end

@testset "boxcox -- exact inverse recovers original series" begin
    x = abs.(randn(50)) .+ 1.0
    for lambda in [-0.5, 0.0, 0.5, 1.0, 1.5]
        y = boxcox(x, lambda)
        @test isapprox(boxcox_inv(y, lambda), x; atol=1e-8)
    end
end

@testset "guerrero_lambda -- bounded, deterministic on a known series" begin
    x = abs.(randn(96)) .+ 5.0
    lam = guerrero_lambda(x, 12)
    @test -1.0 <= lam <= 2.0
end

@testset "guerrero_lambda -- a genuinely heteroskedastic series should NOT select lambda near 1" begin
    # variance growing with the level -- Box-Cox's whole reason to exist;
    # lambda should move meaningfully away from 1 (no transform) toward
    # something that stabilizes variance
    t = 1:96
    x = (1 .+ 0.1 .* t) .* (1 .+ 0.2 .* randn(96))
    lam = guerrero_lambda(x, 12)
    @test !isapprox(lam, 1.0; atol=0.1)
end
```

---

## 4. What to do with this

1. Implement `periodogram`/`spectral_density` in a new
   `src/spectral.jl`, reusing an existing FFT dependency if one already
   exists elsewhere in the project rather than adding a new one.
2. Implement `boxcox`/`boxcox_inv`/`guerrero_lambda` in a new
   `src/transforms.jl`, reusing Stage 4.1's bounded optimizer for the
   lambda search.
3. Run the tests in section 3 -- the periodogram peak-detection test is
   the highest-value one, since it's real, execution-verified ground
   truth, not a structural check alone.
4. **Independently verify Guerrero's method against real `forecast`
   output once CRAN or an equivalent source becomes reachable** -- this
   is the one piece in this handoff genuinely resting on documentation
   alone, flagged explicitly rather than treated as equivalent to the
   periodogram's execution-backed verification.
5. Update `development-sequence.md`'s Chapter Two table: add 1.5 and
   1.6, and record both books' corroboration explicitly so the
   "why here, why now" reasoning isn't lost.
