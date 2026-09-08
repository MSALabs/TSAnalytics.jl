# Handoff: `index.md` (Home Page) Content

For a fresh Claude Code session picking this up with no prior context.
**This is a deliberate, scoped exception to the skeleton handoff's "no
content" rule** — the home page is real content, requested directly,
not part of the deferred chapter-writing work. Every other page in the
skeleton stays a stub; this one page is written now.

## Where this fits

- **Depends on:** the skeleton handoff (`docs-restructure-skeleton-handoff.md`)
  being complete, so the navigation links this page points to actually
  resolve.
- **Model directly, section by section**: `SeasonalAdjustment.jl`'s
  real home page (fetched directly this session). TSAnalytics.jl's
  version mirrors its *structure* exactly, but several sections are
  genuine **mirror-images** in content, not copies — most importantly,
  where the sibling package states plainly *"it does not reimplement
  seasonal adjustment,"* this package's honest equivalent is the
  opposite claim: it does reimplement, and the trust question is
  "verified how," not "wraps what."

---

## Section-by-section content plan

### 1. Opening statement

```markdown
# TSAnalytics.jl

TSAnalytics.jl is a native Julia package for time series analysis --
ARMA and ARIMA models, the GARCH family for volatility, state-space
methods and the Kalman filter, regression with ARIMA errors, and the
diagnostic and decomposition tools that sit around all of them.
```

### 2. Motivating paragraph

```markdown
Fitting a seasonal ARIMA model, checking whether a series has a unit
root, or estimating a GARCH(1,1) are all common, well-understood tasks
-- and until now, Julia has had no single package bringing them
together with the rigor a production statistical workflow needs. Each
piece of this package was built against a primary source (Hamilton;
Durbin & Koopman; Tsay; Shumway & Stoffer) and checked, wherever
possible, against real R and Python output -- not assumed correct
because the formula looked right on paper.
```

### 3. Real, runnable code example

```markdown
```julia
julia> using TSAnalytics

julia> y = dataset("airline")
144-element Vector{Float64}:
 112.0
 118.0
 ⋮

julia> m = auto_arima(y; seasonal=true, m=12)
SarimaModel
  Order:        (0, 1, 1)(0, 1, 1)[12]
  Method:       :ml
  Log-lik:      [value]
  AIC:          [value]

julia> plot(diagnostic_plot(residuals(m), m))
```
![Diagnostic plot](assets/getting-started-diagnostics.png)
```
**Flagged directly, not glossed over**: the exact log-likelihood/AIC
values and the ARIMA order shown above are illustrative placeholders,
not yet re-verified against a real run this session. Whoever writes
this page must run `auto_arima(dataset("airline"); seasonal=true, m=12)`
for real and paste the actual output -- do not ship invented numbers on
the home page of a project this rigorous about verification everywhere
else. The `dataset("airline")` call itself is real (confirmed from
this project's own Stage 3 dataset-API work); only the displayed
numbers need a fresh, real run.

### 4. The architectural statement -- the honest mirror-image of the sibling package's own claim

```markdown
**This package does reimplement these methods natively in Julia.**
That is a deliberate choice, not a shortcut avoided: every stage was
built from a primary source first, then checked against real R and
Python execution wherever those tools could be reached, with every
disagreement between references investigated and resolved rather than
picked arbitrarily. Where verification depth differs -- and it
honestly does, stage to stage -- that is stated directly in this
documentation, not smoothed over.
```

### 5. Status

```markdown
!!! note "Status"
    Stages covering primitives, diagnostics, decomposition, ARMA/ARIMA/
    SARIMA, the GARCH family, state-space methods and the Kalman filter,
    and regression with ARIMA errors (ARIMAX/SARIMAX, both the `:mle`
    and `:tvss` estimation paths) are complete. Vector autoregression,
    cointegration, structural time series, and the companion
    TSFeatures.jl package are planned but not yet started.
```
**Verify this exact stage list against `development-sequence.md`
directly before publishing** — this handoff's author has not re-checked
the live roadmap file this session; state precisely what's actually
done, not what's remembered as done.

### 6. Alpha release

```markdown
!!! warning "Alpha release"
    TSAnalytics.jl is not yet on Julia's General registry.

    ```julia
    using Pkg
    Pkg.add(PackageSpec(url = "https://github.com/MSALabs/TSAnalytics.jl",
                         rev = "v0.1.0-alpha.1"))
    ```
```
**Simpler than the sibling package's own version** — SeasonalAdjustment.jl
needs the joint two-package install because it *depends on*
TSAnalytics.jl; TSAnalytics.jl has no such dependency of its own, so a
single `PackageSpec` is correct here, not an oversight.

### 7. Resources for getting started

```markdown
- Read [Installation and First Check](getting-started/01-installation.md),
  then [Your First Model](getting-started/02-first-model.md).
- New to time series analysis? Start with
  [Why Model Time Series At All?](introduction/01-why-model-time-series.md)
  -- the Introduction is written to be readable without Julia in front
  of you.
- Coming from R or Python? The Manual's
  [translation page](manual/11-coming-from-r-python.md) maps common
  workflows directly.
- Already know the task? The [Manual](manual/01-primitives.md) is
  organized around "how do I ..." questions, not a walkthrough.
```

### 8. Help us improve

Same pattern as the sibling page, pointing at
`https://github.com/MSALabs/TSAnalytics.jl/issues`.

### 9. How the documentation is structured

Adapt the sibling's four-paragraph explanation directly to this
package's own four sections (Getting Started / Manual / Introduction /
API Reference), using the actual chapter lists from the skeleton
handoff so the descriptions are concrete, not generic.

### 10. Citing

```markdown
```bibtex
@software{TSAnalyticsJL,
    author  = {{XKDR Forum}},
    title   = {{TSAnalytics.jl}: Native time series analysis in Julia},
    year    = {2026},
    url     = {https://github.com/MSALabs/TSAnalytics.jl}
}
```

Work leaning on a specific method should also cite its primary source
directly -- Hamilton (1994) for ARMA/state-space theory, Durbin &
Koopman (2012) for the Kalman filter and diffuse initialization, Tsay
(2010) for the GARCH family. The
[Further Reading](introduction/B-further-reading.md) appendix carries
the complete list.
```
**Confirm the exact publication years above before publishing** — cited
from general knowledge in this handoff, not re-verified against the
actual title pages this session, unlike numbers pulled from real
execution elsewhere in this project.

### 11. Note on the org move

Match the sibling page's own note (repository moving to `xKDR` in due
course) if TSAnalytics.jl is genuinely part of the same planned move --
**confirm this directly rather than assume**, since it's a factual claim
about organizational plans, not something inferable from the codebase.

### 12. About XKDR Forum

```markdown
TSAnalytics.jl is developed at [XKDR Forum](https://xkdr.org), a
non-profit research organisation based in Mumbai, India. It is the
foundation the rest of this project family is built on --
[SeasonalAdjustment.jl](https://github.com/MSALabs/SeasonalAdjustment.jl)
depends on it directly for exactly this reason, rather than
duplicating primitives that belong here.
```

### 13. License

```markdown
TSAnalytics.jl is licensed under the [MIT licence](LICENSE).
```
Simpler than the sibling page's license section -- no bundled
third-party binary here, so no carve-out is needed.

---

## What to do with this

1. Write `docs/src/index.md` following the twelve sections above, in
   order, matching the sibling page's tone and structure exactly.
2. **Run the section 3 code example for real** and replace the
   placeholder output with actual values -- this is the one piece of
   this handoff that must not ship as written here.
3. **Verify the Status list (section 5) against the current, real
   `development-sequence.md`** before publishing, not from memory.
4. **Confirm the org-move note (section 11) and publication years
   (section 10) directly** rather than carry them over unverified.
5. Everything else in the skeleton stays a stub -- do not let this
   home-page content task expand into writing other chapters as well.
