# Autoregressive Errors with Changing Variance

The last chapter of Part VII, and the one that joins it back to Part
V. A regression can have errors that are correlated, Chapter 34's
subject, and a variance that moves, Chapter 24's. Real series often
have both problems in the same series, and this chapter fits them
together rather than one after the other.

## Two problems in one series

```@example ch37
using TSAnalytics, Plots, Statistics, LinearAlgebra

dc = dataset("cmort"); dt = dataset("tempr")
y = dc.value; x = dt.value
n = length(y)
X = reshape(x, :, 1)
Xd = hcat(X, ones(n))
beta_ols = Xd \ y
resid_ols = y .- Xd*beta_ols

p1 = plot(resid_ols; title="OLS residuals", legend=false)
p2 = plot(acf(resid_ols, 1:20); title="their ACF")
p3 = plot(acf(resid_ols.^2, 1:20); title="their squared ACF")
plot(p1, p2, p3; layout=(1,3), size=(950,280))
println("resid ACF, lags 1-3:   ", round.(acf(resid_ols,1:3).values,digits=3))
println("resid^2 ACF, lags 1-3: ", round.(acf(resid_ols.^2,1:3).values,digits=3))
```

Chapter 34's own series, and both problems are visible in the same
three panels. The residuals are autocorrelated — `0.52` at lag one,
Chapter 34's own finding. Their squares are autocorrelated too — `0.51`
at lag one, decaying more slowly than a coincidence would. Chapter
12's diagnostic panel would fail this series on two separate counts at
once, and that is the normal state of weekly mortality data, not a
constructed example.

## Fixing one at a time

```@example ch37
m_ar = fit_arimax(y, (1,0,0), X)
nu = y .- Xd*m_ar.beta
e_ar = Vector{Float64}(undef, n); e_ar[1] = nu[1]
for t in 2:n
    e_ar[t] = nu[t] - m_ar.arma.ar[1]*nu[t-1]
end
std_ar = e_ar ./ sqrt(m_ar.arma.sigma2)
println("AR(1)-errors model: phi=", round(m_ar.arma.ar[1],digits=3))
println("  filtered residual ACF, lag 1:   ", round(acf(std_ar,1:1).values[1],digits=3), "  (raw regression residual was 0.52)")
println("  filtered residual^2 ACF, lag 1: ", round(acf(std_ar.^2,1:1).values[1],digits=3))
```

Chapter 34's model, applied and checked rather than only fitted. The
AR(1) filter does real work — the raw regression residual's own
lag-one correlation was `0.835` here (this filtered residual is a
different quantity from the OLS residual above; the joint fit's own
regression coefficient is `0.2424`, not OLS's `-0.4866`, exactly
Chapter 34's sign-reversal result) — but it does not fall to zero. A
single AR(1) term captures most, not all, of a real series' own
persistence, and the honest number, `-0.34`, is a real and
statistically significant remainder rather than noise on `508`
observations. **What an AR(1) filter cannot touch at all is the
variance clustering** — the squared-residual correlation is
essentially unchanged by fixing the mean equation, `0.17` against the
original `0.51`, better only because some of the level persistence
that inflated both numbers together is now gone, not because anything
here addressed volatility directly.

```@example ch37
m_g = fit_garch(resid_ols, 1, 1)
std_g = m_g.resid ./ sqrt.(m_g.sigma2)
println("GARCH(1,1)-on-OLS-residuals: omega=", round(m_g.omega,digits=2), " alpha=", round(m_g.alpha[1],digits=3),
        " beta=", round(m_g.beta[1],digits=3))
println("  standardised residual ACF, lag 1:   ", round(acf(std_g,1:1).values[1],digits=3))
println("  standardised residual^2 ACF, lag 1: ", round(acf(std_g.^2,1:1).values[1],digits=3))
```

The other direction: Chapter 25's own model, fitted directly to the
OLS residuals with no AR filtering at all. The squared-residual
correlation is essentially gone — `0.001` at lag one — GARCH does
exactly what it is built to do. The plain residual correlation is
untouched, `0.37`, because nothing in a GARCH variance equation has
any way to touch the mean. Each fix addresses exactly one of the two
problems and is silent about the other, and running them as two
separate steps — the two-step habit Chapter 34 already warned against
— estimates each stage under an assumption the other stage
contradicts.

## Both at once

```@example ch37
m_comb = fit_autoreg_garch(y, 1, X; garch_order=(1,1))
println(m_comb)
```

The construction is a chain, and each link is something already met:

```
nu_t = y_t − x_t·beta                          (regression residual, Chapter 34)
e_t  = nu_t − phi·nu_{t−1}                      (AR innovation, Chapter 34)
h_t  = omega + alpha·e_{t−1}² + garch_beta·h_{t−1}   (GARCH variance, Chapter 25)
loglik = −0.5 Σ [ log(2π h_t) + e_t²/h_t ]
```

`beta`, `phi`, and the GARCH variance parameters are estimated
**jointly**, in one optimisation, rather than fitted in stages that
silently disagree with each other.

```@example ch37
std_comb = m_comb.garch.resid ./ sqrt.(m_comb.garch.sigma2)
plot(sqrt.(m_comb.garch.sigma2); label="conditional std. dev.", linewidth=1.5)
plot!(twinx(), std_comb; label="standardised residual", color=:orange, alpha=0.6, legend=:topright,
      title="both problems, addressed together")
println("standardised residual ACF, lag 1:   ", round(acf(std_comb,1:1).values[1],digits=3))
println("standardised residual^2 ACF, lag 1: ", round(acf(std_comb.^2,1:1).values[1],digits=3))

lb = ljungbox_test(std_comb, 10)
alm = arch_lm_test(std_comb, 5)
println("Ljung-Box (10 lags):  stat=", round(lb.statistic,digits=2), "  p=", round(lb.pvalue,digits=4))
println("ARCH-LM (5 lags):     stat=", round(alm.statistic,digits=2), "  p=", round(alm.pvalue,digits=4))
```

Report this plainly rather than the tidy version. The squared-residual
correlation is now effectively zero, and **ARCH-LM improves by more
than an order of magnitude** against the AR-only model's own `p =
0.0002` — the combined model's variance equation is doing genuine
work. **Ljung-Box does not pass** — `p < 0.0001` — because a single
AR(1) term was never a complete description of this series' own mean
structure to begin with, on `508` real observations, and nothing a
GARCH variance equation does can repair that. This is worth sitting
with rather than glossing over: **the combined construction fixes
exactly the problem it targets and no other** — the variance equation
cannot rescue an underspecified mean equation, the same way an AR
filter earlier could not touch variance clustering. A cleaner mean
specification (a second AR lag, an MA term) is a separate decision,
answered by the same diagnostics used here, not a reason to distrust
the GARCH half of what was actually fixed.

## The term that was nearly dropped

The `disagreement` box — and unusually, the disagreement is with a
shortcut, not another package.

```@example ch37
function naive_loglik(beta, phi, sigma2, yv, Xmat)
    nu = yv .- Xmat*beta
    ll = -0.5*(log(2pi) + log(sigma2) + nu[1]^2/sigma2)
    for t in 2:length(yv)
        e = nu[t] - phi*nu[t-1]
        ll += -0.5*(log(2pi) + log(sigma2) + e^2/sigma2)
    end
    return ll
end
function correct_loglik(beta, phi, sigma2, yv, Xmat)
    nu = yv .- Xmat*beta
    statvar = sigma2/(1-phi^2)
    ll = -0.5*(log(2pi) + log(statvar) + nu[1]^2/statvar)
    for t in 2:length(yv)
        e = nu[t] - phi*nu[t-1]
        ll += -0.5*(log(2pi) + log(sigma2) + e^2/sigma2)
    end
    return ll
end

ll_naive = naive_loglik(m_ar.beta, m_ar.arma.ar[1], m_ar.arma.sigma2, y, Xd)
ll_correct = correct_loglik(m_ar.beta, m_ar.arma.ar[1], m_ar.arma.sigma2, y, Xd)
println("shortcut (nu[1] treated as an ordinary innovation): ", round(ll_naive,digits=6))
println("correct (nu[1] drawn from its own stationary variance): ", round(ll_correct,digits=6))
println("chapter 34's own verified joint fit:  ", round(m_ar.loglik,digits=6))
println("gap, shortcut vs correct: ", round(ll_naive - ll_correct, digits=4))
```

An AR(1) error process does not start from nowhere. Its very first
value is itself a draw from the process's own stationary distribution
— variance `σ²/(1−φ²)`, larger than the ordinary innovation variance
`σ²` because a persistent process wanders further than one step's
worth of noise would suggest. Conditioning on that first value, or
filtering it as though an ordinary lag were available, throws that
term's genuine information away.

On this series the gap is `0.25` log-likelihood units, and its
**sign** is itself instructive: the correct treatment here scores
*higher* than the shortcut, not lower. Whether the shortcut over- or
understates the likelihood is not fixed in general — it depends on how
far the series' very first residual happens to land from zero, which
is a property of the specific data, not of the method. What is fixed
is that the shortcut is *wrong*, by an amount that does not vanish as
the sample grows, because it is a single mis-specified term rather
than an estimation-noise effect that averages out.

```@example ch37
function gap(phi, nu1, sigma2)
    statvar = sigma2/(1-phi^2)
    correct = -0.5*(log(2pi)+log(statvar)+nu1^2/statvar)
    naive = -0.5*(log(2pi)+log(sigma2)+nu1^2/sigma2)
    return naive - correct
end
nu1 = (y .- Xd*m_ar.beta)[1]
phis = 0.0:0.02:0.995
gaps = [gap(p, nu1, m_ar.arma.sigma2) for p in phis]
plot(phis, gaps; xlabel="phi", ylabel="shortcut minus correct", linewidth=2,
     title="the correction, swept across phi", legend=false)
println("gap at phi=0.3: ", round(gap(0.3,nu1,m_ar.arma.sigma2),digits=4),
        "   at phi=0.834 (fitted): ", round(gap(0.834,nu1,m_ar.arma.sigma2),digits=4),
        "   at phi=0.99: ", round(gap(0.99,nu1,m_ar.arma.sigma2),digits=4))
```

`1/(1−φ²)` diverges as `φ→1`, and the sweep shows it directly — the
gap's magnitude grows without bound as `φ` approaches one, though not
monotonically on the way there for this particular `nu[1]`: the two
competing terms in the gap (a log-variance term that always grows, and
a quadratic term that can shrink) trade off differently depending on
how large the realised first residual is, so the curve can cross zero
before the divergence takes over. What does not change is the
practical lesson — **an approximation whose error depends on the very
parameter being estimated is dangerous precisely because its size
cannot be judged in advance**, only after the fact, the way this
sweep just did.

```@example ch37
ll_internal, _, _, ok = TSAnalytics._ar_garch_loglik(m_ar.beta, [m_ar.arma.ar[1]], m_ar.arma.sigma2,
                                                       Float64[0.0], Float64[0.0], y, Xd, 1)
println("combined model's own machinery, GARCH degenerate (alpha=garch_beta=0): ", ll_internal)
println("chapter 34's target: ", m_ar.loglik)
println("agreement to: ", abs(ll_internal - m_ar.loglik))
```

This is the chapter's verification, and it is worth naming the
technique plainly: with no reference implementation covering this
exact combination — checked directly, Python's `arch.univariate.ARX`
joins a series' own lags with regressors and a volatility process,
which is a different model from regression-then-AR-structured-errors,
and SAS's `PROC AUTOREG` is the closest single reference and is not
runnable here — correctness is established by collapsing the general
model onto an already-trusted special case. Chapter 15 used this
technique for STL's own edge behaviour. Chapter 32 used it for the
time-varying reduction test. Setting the GARCH parameters to exactly
zero forces the combined likelihood back onto Chapter 34's own
already-verified plain fit, and the two agree to machine precision —
confirming, incidentally, that [`fit_autoreg_garch`](@ref) already
implements the stationary first-observation treatment correctly
throughout, not merely in the hand-derived check above; the shortcut
shown was a hazard this chapter constructed by hand to demonstrate,
not a bug this package's own fitting function contains.

!!! india "The Indian Series"
    Chapter 17 already established that monthly Indian inflation and
    industrial-growth series typically carry AR coefficients between
    `0.7` and `0.95` — genuinely persistent, not the mild `φ ≈ 0.3`
    where this chapter's own correction is smallest. Read off the
    sweep above at `φ = 0.9`: the stationary variance is more than
    five times the ordinary innovation variance, and the omitted term
    is correspondingly large. A regression-with-AR-errors
    implementation that takes the conditioning shortcut will be
    systematically wrong on precisely the persistent series that
    Indian macroeconomic work is built on — and, per the point just
    made about the gap's sign, wrong in whichever direction that
    series' own first residual happens to produce, with no warning
    that anything is off.

## What it buys

```@example ch37
h = m_comb.garch.sigma2
window = 60:190
const_sd = sqrt(mean(m_comb.garch.resid.^2))
plot(window, m_comb.garch.resid[window]; label="AR innovation", color=:gray, alpha=0.5)
plot!(window, 1.96 .* fill(const_sd, length(window)); label="constant-variance 95% band", linestyle=:dash, color=:blue)
plot!(window, -1.96 .* fill(const_sd, length(window)); label="", linestyle=:dash, color=:blue)
plot!(window, 1.96 .* sqrt.(h[window]); label="combined model 95% band", color=:red, linewidth=2)
plot!(window, -1.96 .* sqrt.(h[window]); label="", color=:red, linewidth=2,
      title="a genuinely turbulent stretch")
println("mean conditional variance: ", round(mean(h),digits=1),
        "   peak: ", round(maximum(h),digits=1), " at week ", argmax(h),
        "   constant-variance model's own single value: ", round(const_sd^2,digits=1))
```

Chapter 24 made exactly this point about GARCH alone; it applies here
to a regression model just as directly. Across the window shown, the
conditional variance swings from close to its long-run average up to
nearly five times that average — the constant-variance band is too
wide through the calm stretches and too narrow exactly where it
matters, through the turbulent one. The combined model's own band
tracks the conditions instead of averaging over them.

The honest cost: more parameters, a harder joint optimisation, and
convergence that is not guaranteed on every series — `fit_autoreg_garch`
takes several restarts by default for exactly this reason. For a
series where Chapter 11's own ARCH-LM test does not reject constant
variance in the first place, none of this machinery is worth adding,
and that test is how you find out before committing to it.

## Where this leaves you

Part VII is finished. Outside information — a measured regressor, a
coefficient allowed to drift, a calendar, and now errors that are both
correlated and heteroskedastic at once — can all be brought into a
single model, each piece checked on its own terms rather than assumed
to work because the pieces are each individually familiar.

Part VIII turns to what national statistical offices actually do with
all of this — a different problem, with a different standard of
evidence than a single verified fit.
