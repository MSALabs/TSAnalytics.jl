# Handoff: Stage 4.2 — Stationarity/Invertibility Reparametrization (Monahan 1984)

Status: **done.** `partrans`/`invpartrans` (`src/monahan.jl`,
`test/test_monahan.jl`) built exactly per section 5's design and
translation. Section 3's exact numbers were independently re-derived
(not just transcribed) via a from-scratch Python transcription of R's
actual C source before trusting the Julia 1-indexed translation, per
section 5's own warning that this exact category of off-by-one has
been the real bug source elsewhere in this project. All of section 6's
hand-verified values pass exactly, plus round-trip and exact-value tests
across `p=0..6` and container-agnostic input. One property worth noting
for future readers testing this function: **only the intermediate PACF
values (`tanh(raw)`) are bounded in `(-1,1)`** -- the final AR/MA
coefficients after the Durbin-Levinson recursion are not (e.g.
`partrans([0.9,-0.8,0.7,-0.6,0.5])`'s first coefficient is `~2.17`),
since Monahan's transform guarantees *stationarity*, a weaker condition
than "each coefficient's magnitude `< 1`" -- an incorrect version of that
bounds assumption was caught in this project's own first test-writing
pass and fixed before it was ever committed as a real bug. Per section 7,
no cross-language numeric comparison against Python's convention was
attempted -- confirmed distinct is the correct assertion, not a match.

For a fresh Claude Code session picking this up with no prior context.
**A real, numerically-confirmed discrepancy was found between R's and
Python's implementations of this transform** -- not a naming or default
difference like most earlier findings, but two genuinely different
mathematical squashing functions. Read section 3 before implementing.

## Where this fits

- **Depends on:** Stage 4.1 (for the end-to-end optimizer check; the
  transform's own correctness can and should be verified independently
  first, without needing an optimizer at all).
- **This is the piece every future MLE-fit AR/MA model depends on** --
  ARIMA, SARIMAX, GARCH all need it. A subtle error here wouldn't crash
  anything; it would silently produce plausible-looking but wrong
  coefficients, which is a worse failure mode than a loud one.
- **Already partially drafted**: the very first message of this whole
  project (the X13ArimaSeats Phase 1 sketch) included a `partrans`
  function for exactly this. That draft used `tanh` -- section 3 below
  confirms this was actually the right call, matching R's real source
  exactly (verified now, not assumed then).

---

## 1. Verified reference: R (`stats::arima`'s C source, `arima.c`)

R's transform is compiled C, not visible via `print()` at the R level --
fetched directly from the well-known `wch/r-source` GitHub mirror of R's
actual source tree. Full function, confirmed:

```c
static void partrans(int p, double *raw, double *new)
{
    int j, k;
    double a, work[100];

    /* Step one: map (-Inf, Inf) to (-1, 1) via tanh
       The parameters are now the pacf phi_{kk} */
    for(j = 0; j < p; j++) work[j] = new[j] = tanh(raw[j]);
    /* Step two: run the Durbin-Levinson recursions to find phi_{j.} */
    for(j = 1; j < p; j++) {
        a = new[j];
        for(k = 0; k < j; k++)
            work[k] -= a * new[j - k - 1];
        for(k = 0; k < j; k++) new[k] = work[k];
    }
}
```

And the inverse (needed for initializing the optimizer from OLS/
Yule-Walker starting estimates), also confirmed from source:

```c
static void invpartrans(int p, double *phi, double *new)
{
    int j, k;
    double a, work[100];
    for(j = 0; j < p; j++) work[j] = new[j] = phi[j];
    /* Run the Durbin-Levinson recursions backwards */
    for(j = p - 1; j > 0; j--) {
        a = new[j];
        for(k = 0; k < j; k++)
            work[k] = (new[k] + a * new[j - k - 1]) / (1 - a * a);
        for(k = 0; k < j; k++) new[k] = work[k];
    }
    for(j = 0; j < p; j++) new[j] = atanh(new[j]);
}
```

**This exactly matches the X13ArimaSeats Phase 1 sketch's `partrans`**
(tanh, then the same in-place Durbin-Levinson recursion structure) --
that earlier draft was correct, now confirmed against real R source
rather than general familiarity with the method.

## 2. Verified reference: Python (`statsmodels.tsa.statespace.tools`, pure Python, directly inspected)

```python
def constrain_stationary_univariate(unconstrained):
    """... References: Monahan, John F. 1984. ..."""
    n = unconstrained.shape[0]
    y = np.zeros((n, n), dtype=unconstrained.dtype)
    r = unconstrained / ((1 + unconstrained**2) ** 0.5)     # <-- NOT tanh
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i - 1]
        y[k, k] = r[k]
    return -y[n - 1, :]                                       # <-- negated

def unconstrain_stationary_univariate(constrained):
    n = constrained.shape[0]
    y = np.zeros((n, n), dtype=constrained.dtype)
    y[n-1:] = -constrained
    for k in range(n-1, 0, -1):
        for i in range(k):
            y[k-1, i] = (y[k, i] - y[k, k]*y[k, k-i-1]) / (1 - y[k, k]**2)
    r = y.diagonal()
    x = r / ((1 - r**2) ** 0.5)
    return x
```

Both functions **cite the identical Monahan (1984) reference** R's
implementation is based on -- same paper, genuinely different squashing
function chosen to implement it.

## 3. The discrepancy, verified numerically -- not a guess

Same raw unconstrained input `[0.5, -0.3, 0.2]`, fed through both
transcribed algorithms:

```
R-style (tanh):                       [ 0.65423563, -0.40909391,  0.19737532]
Python-style (u/sqrt(1+u^2), negated): [-0.26235416,  0.22484416, -0.19611614]
```

**Completely different constrained coefficients** -- different
magnitudes, mostly different signs. Both were verified to independently
satisfy the stationarity condition (AR polynomial roots outside the unit
circle) -- **both are individually correct implementations of "a"
Monahan-style transform, they just parameterize the same constrained
space differently.**

**What this does and does not mean, precisely:**
- It does **not** mean either implementation has a bug. `tanh` and
  `u/sqrt(1+u^2)` are both smooth, odd, monotonic bijections from R onto
  `(-1,1)` -- either is a mathematically valid choice for "map an
  unconstrained real to a partial autocorrelation."
- It **does** mean: the *raw, unconstrained optimizer-space parameter
  values* are not comparable between an R-based and a Python-based
  implementation, even fitting the identical model to identical data.
  Two correctly-converged optimizers using different transforms will
  generally sit at different points in raw parameter space.
- It should **not** affect the *final fitted AR/MA coefficients* --
  those are determined by where the likelihood is actually maximized in
  the (transform-independent) constrained coefficient space, not by
  which path the optimizer took to get there. This is the number that
  matters for validating a fitted model's output; the raw parameters are
  a pure implementation detail.
- **Practical implication**: don't attempt to cross-validate this
  package's *raw* optimizer parameters against R's or Python's internals
  step-by-step during optimization -- validate the *final fitted
  coefficients* instead, which is what every other stage's testing
  approach in this project already does.

## 4. Design decision: adopt R's tanh-based convention

Reasons, stated explicitly rather than defaulted into:
1. **Already partially implemented and reasoned through** in this
   project's own Phase 1 sketch -- lower risk to build on existing,
   now-verified work than switch conventions.
2. **Simpler**: no extra negation step to explain and get right; `tanh`
   is a more universally recognizable squashing function than
   `u/sqrt(1+u^2)`.
3. Either choice is equally defensible mathematically (section 3) -- this
   is a case where there's no "more correct" answer, so picking the
   simpler, already-drafted option is the reasonable tiebreaker.

Document this choice explicitly in the function's docstring, including
the "don't expect raw-parameter cross-language matching" caveat from
section 3 -- this is exactly the kind of thing worth writing down once
rather than having someone rediscover confusedly later.

---

## 5. Implementation

```julia
"""
    partrans(raw::AbstractVector{<:Real}) -> Vector{Float64}

Map unconstrained real parameters (as used by the optimizer) to AR/MA
coefficients guaranteed to lie in the stationary/invertible region, via
the Monahan (1984) transform: tanh to squash each raw parameter into a
partial autocorrelation in (-1,1), then the Durbin-Levinson recursion to
convert those PACF values into AR (or MA) coefficients.

Matches R's `stats::arima` C implementation (`arima.c`'s `partrans`)
exactly -- verified against R's actual source, not just the general
method description. Deliberately does NOT match Python's
`statsmodels.tsa.statespace.tools.constrain_stationary_univariate`,
which implements the same Monahan (1984) result via a different
squashing function (`u/sqrt(1+u^2)`, plus a sign negation) -- both are
individually valid; see the Stage 4.2 handoff doc section 3 for why raw
parameter values are not expected to match across the two conventions,
and why that doesn't affect final fitted coefficients.
"""
function partrans(raw::AbstractVector{<:Real})
    p = length(raw)
    p == 0 && return Float64[]
    new = tanh.(collect(Float64, raw))
    work = copy(new)
    for j in 2:p
        a = new[j]
        for k in 1:(j-1)
            work[k] -= a * new[j-k]
        end
        new[1:j-1] .= work[1:j-1]
    end
    return new
end

"""
    invpartrans(phi::AbstractVector{<:Real}) -> Vector{Float64}

Inverse of [`partrans`](@ref): map AR/MA coefficients back to the
unconstrained space, for initializing the optimizer from a
closed-form starting estimate (e.g. OLS or Yule-Walker) rather than an
arbitrary point. Matches R's `invpartrans` (`arima.c`) exactly, verified
against source.
"""
function invpartrans(phi::AbstractVector{<:Real})
    p = length(phi)
    p == 0 && return Float64[]
    new = collect(Float64, phi)
    work = copy(new)
    for j in p:-1:2
        a = new[j]
        for k in 1:(j-1)
            work[k] = (new[k] + a*new[j-k]) / (1 - a^2)
        end
        new[1:j-1] .= work[1:j-1]
    end
    return atanh.(new)
end
```

**Indexing note**: R's C source is 0-indexed (`for(j = 1; j < p; j++)`,
inner `new[j - k - 1]`); the Julia translation above is 1-indexed
(`for j in 2:p`, inner `new[j-k]`). Verify this translation numerically
against section 6's values before trusting it -- this exact category of
off-by-one has been the real bug source in this project before (the
original `_ols` lagged-difference columns, the STL extrapolation
indexing), not the algorithm's math itself.

---

## 6. Hand-verified test values

From the verification script in section 3 (exact, reproducible):

```julia
using Test

@testset "partrans (R tanh convention)" begin
    raw = [0.5, -0.3, 0.2]
    result = partrans(raw)
    @test isapprox(result, [0.65423563, -0.40909391, 0.19737532]; atol=1e-6)

    # round-trip: invpartrans(partrans(raw)) should recover raw
    @test isapprox(invpartrans(result), raw; atol=1e-8)

    # p=0 and p=1 edge cases
    @test partrans(Float64[]) == Float64[]
    @test isapprox(partrans([0.5]), [tanh(0.5)]; atol=1e-12)  # p=1: no DL recursion needed
end
```

A stationarity round-trip check (confirming the AR polynomial's roots
land outside the unit circle) is worth adding once this project has a
polynomial-roots utility -- likely arriving naturally with Stage 6's
SARIMA characteristic-polynomial work, not worth a one-off dependency
just for this test.

---

## 7. What to do with this

1. Implement `partrans`/`invpartrans` per section 5, double-checking the
   0-indexed-to-1-indexed translation specifically against section 6's
   numbers.
2. Run the tests in section 6.
3. **Do not** attempt to also cross-validate against Python's convention
   numerically -- per section 3, that's expected to disagree and isn't a
   bug. If a test comparing against Python's transform is ever written,
   it should assert they're *different*, not that they match (the same
   pattern already used in the Stage 1.3 ACF/PACF handoff for the
   R-vs-Python PACF default divergence).
4. Update `development-sequence.md`'s Stage 4.2 row: mark implemented,
   and make sure the "R vs Python convention, deliberately not matching"
   finding is visible there too, not just in this handoff -- it's
   exactly the kind of thing someone skimming the roadmap later could
   otherwise mistake for an unnoticed bug.
