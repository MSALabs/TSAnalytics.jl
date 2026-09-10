# Why Variance Changes

Every model in Part IV assumed the variance of the noise was a fixed
number, quietly, without ever saying so out loud. Chapter 11 showed a
series where that assumption was plainly false and nothing built since
has done anything about it. This chapter picks the thread back up.

## The residuals that passed

```@example ch24
using TSAnalytics, Plots, Random

Random.seed!(20)
n = 400
e = randn(n)
resid = zeros(n)
resid[1] = e[1]
for t in 2:n
    resid[t] = sqrt(0.05 + 0.85 * resid[t-1]^2) * e[t]
end
lb = ljungbox_test(resid, 10)
plot(resid; title="Chapter 11's residuals again -- Ljung-Box p = $(round(lb.pvalue, digits=3))", legend=false)
```

The exact same **simulated** residuals Chapter 11 closed on. Thirteen
chapters of tooling have now been pointed at this series, and every one
of them says it is fine — the correlogram is clean, the portmanteau
test passes at `p = 0.932`, nothing here looks obviously wrong by any
standard from Part II. And the picture still shows quiet stretches
followed by violent ones. Something in this series is predictable, and
nothing built so far can predict it.

## Prices and returns

```@example ch24
d = dataset("nyse")
r = d.value
price = cumprod(1 .+ r)  # reconstructed relative price path -- exact, since return[t] determines price[t]/price[t-1]
p1 = plot(price; title="NYSE, reconstructed relative price level", legend=false)
p2 = plot(r; title="NYSE, daily returns", legend=false)
plot(p1, p2; layout=(2,1), size=(700,450))
```

The bundled series is already returns, not prices — the price panel
above is reconstructed by cumulative product, which recovers the exact
*relative* path (the absolute starting level is not recoverable from
returns alone, so it is anchored arbitrarily at `1`). Chapter 3
established that prices are I(1) and returns are I(0) — differencing
logs *is* the return transformation — so the return series is the
stationary object, and by Part IV's own standards it should now be
tractable: no trend, no seasonality, a mean indistinguishable from
zero.

```@example ch24
p1 = plot(acf(r, 1:30); title="ACF of returns")
p2 = plot(acf(r.^2, 1:30); title="ACF of squared returns")
plot(p1, p2; layout=(1,2), size=(800,300))
```

This is the chapter. The first panel is close to empty — the largest
bar barely clears `0.10`, well within what Chapter 5's own bands would
tolerate at this sample size. The second is not: squared returns carry
real autocorrelation out past lag twenty, several times larger than
anything in the first panel. The *sign* of tomorrow's return is close
to unpredictable, which is what an efficient market implies and what
the empty first panel shows. The *size* is a different question
entirely, and this pair of panels is the only evidence needed that it
has a different answer.

## What clustering looks like

```@example ch24
rolling_sd(v, w) = [sqrt(sum(abs2, v[max(1,i-w+1):i]) / w) for i in w:length(v)]
plot(rolling_sd(r, 60); title="NYSE returns, 60-day rolling standard deviation", legend=false)
```

It moves, and it moves persistently rather than jittering randomly —
from a low of about `0.0049` to a high of about `0.0347`, a sevenfold
range, over windows wide enough that sampling noise in the *estimate*
would be small relative to a swing this large. Calm stretches last
months; turbulent stretches last months. This is the thing itself
changing, not noise in how it is measured.

```@example ch24
using Statistics
mu, sig = mean(r), std(r)
kurt = mean(((r .- mu) ./ sig).^4) - 3
skew = mean(((r .- mu) ./ sig).^3)
println("excess kurtosis: ", round(kurt, digits=1), "   skewness: ", round(skew, digits=2))
sorted_r = sort(r)
n3 = length(sorted_r)
theoretical_q = [TSAnalytics._std_normal_quantile((i - 0.5) / n3) for i in 1:n3]
p1 = histogram(r; bins=80, normalize=true, legend=false, title="return distribution")
xs = range(minimum(r), maximum(r); length=200)
plot!(p1, xs, [exp(-(x-mu)^2/(2sig^2))/(sig*sqrt(2π)) for x in xs]; linewidth=2, color=:red)
p2 = scatter(theoretical_q, sorted_r; markersize=2, legend=false, title="Q-Q plot")
lo, hi = extrema(theoretical_q)
plot!(p2, [lo,hi], [lo,hi]; color=:red, linestyle=:dash)
plot(p1, p2; layout=(1,2), size=(800,320))
```

Excess kurtosis of `63.5` — a wildly fat-tailed distribution by any
normal-theory standard, though a fair share of that specific number
belongs to one single day: the return series includes 19 October 1987
("Black Monday"), an `18`-standard-deviation move that would be
essentially impossible under normality and is a matter of public
record under it. Even setting that one day aside, the Q-Q plot's tails
depart from the reference line well before either extreme — real,
routine fat-tailedness, not an artefact of one crash. Chapter 11
explained why this matters: prediction intervals built from normal
quantiles will be too narrow, with worse coverage than advertised,
most exactly where being wrong is most expensive.

Worth noting something not obvious on first look: a mixture of normal
distributions with *different* variances is itself heavy-tailed, even
though each individual piece is perfectly normal. If the variance
changes over time and everything is pooled together into one
histogram, fat tails appear automatically, with no separate mechanism
required. **The non-normality in the left panel and the clustering in
the previous section's ACF may be the same phenomenon, seen twice** —
worth realising now, before Chapter 25 starts building a model for one
of them.

## Not every variance problem is clustering

```@example ch24
using DelimitedFiles
Random.seed!(31)
r_shift = vcat(0.01 .* randn(400), 0.05 .* randn(400))
y_air = Float64.(readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)[:, 2])
d_air = diff(y_air)

p1 = plot(r; title="clustering (NYSE)", legend=false)
p2 = plot(r_shift; title="a one-off level shift (simulated)", legend=false)
p3 = plot(y_air; title="variance growing with the level (AirPassengers)", legend=false)
plot(p1, p2, p3; layout=(3,1), size=(700,600))
```

Three series, three different variance behaviours, three different
remedies. The third was already solved in Chapter 7 by a
transformation and needs nothing further here. The second is a
structural break — the **simulated** series above literally switches
from `σ = 0.01` to `σ = 0.05` partway through and stays there — and it
wants an intervention term or a split sample, not a variance model.
Only the first is genuinely what Part V is about: variance that rises
and falls repeatedly around some level, never settling.

```@example ch24
for (nm, series) in [("clustering (NYSE)", r), ("level shift", r_shift), ("growing with level (ΔAirPassengers)", d_air)]
    a = arch_lm_test(series)
    dk = dk_heteroskedasticity_test(series)
    println(nm, ":")
    println("  ARCH-LM:  stat=", round(a.statistic, digits=2), "  p=", round(a.pvalue, digits=4))
    println("  DK ratio: F=", round(dk.statistic, digits=3), "  p=", round(dk.pvalue, digits=4))
end
```

Reported as they actually came out, not tidied to fit the story: **all
three reject ARCH-LM decisively.** A one-off shift and a steadily
growing variance both produce squared-residual autocorrelation just as
real clustering does, for an unsurprising reason — a low-variance block
followed by a high-variance block looks, to a test built on squared
autocorrelation, similar to variance that has genuinely clustered.
ARCH-LM alone does not distinguish the three cases here, and reporting
that plainly is more useful than pretending it does.

The Durbin–Koopman variance-ratio test's **F-statistic**, not just its
p-value, is where the distinction actually shows up: `1.313` for the
genuinely clustering series against `22.74` for the level shift and
`9.21` for the steadily growing one. All three are "significant" at
this sample size, but the shift and the trend produce a first-half-
versus-second-half variance ratio many times larger than clustering
ever does, because clustering wanders back and forth rather than
moving in one direction. Reading the *size* of the statistic, not only
whether it clears a threshold, is what actually separates these three
cases in practice.

## Why anyone cares

```@example ch24
idx = 788:1088  # a stretch of NYSE data spanning the October 1987 crash
r_window = r[idx]
overall_sd = std(r)
rs = rolling_sd(r_window, 30)
z = 1.96
p1 = plot(r_window; ribbon=(z*overall_sd, z*overall_sd), title="constant-variance interval", legend=false)
p2 = plot(r_window[30:end]; ribbon=(z .* rs, z .* rs), title="time-varying interval (60-day rolling)", legend=false)
plot(p1, p2; layout=(2,1), size=(700,500))
```

A stretch of NYSE data spanning the October 1987 crash, with two kinds
of interval drawn around it. The constant-variance interval, built from
the series' own overall standard deviation, is far too wide through
most of the calm stretch and is blown straight through the moment the
crash actually happens — wrong in both directions, and right only on
average, which is the least useful place an interval can be right. A
time-varying interval, built from nothing more sophisticated than a
rolling window here, narrows through the calm periods and widens
through the turbulent one, tracking what actually happened rather than
a single number computed once over the whole sample.

State the stakes concretely: a risk limit set from a constant-variance
model is too loose exactly when markets are turbulent — which is
exactly when a risk limit is supposed to matter. That is not an
academic concern. It is the reason this entire literature exists and
the reason financial regulators require variance models of the kind
the next three chapters build, not the kind built so far.

!!! india "The Indian Series"
    Indian equity and currency series show both patterns from this
    chapter's fourth section, and telling them apart matters in
    practice. There is genuine clustering of the kind the rest of Part
    V models. There are also datable variance shifts tied to specific
    policy moments — the 1991 liberalisation, the 2016 demonetisation
    announcement, the introduction of currency futures — where the
    variability changed level once and stayed changed rather than
    reverting.

    Fitting a clustering model to what is really a structural break
    produces a model that persistently over-predicts volatility in the
    calm regime and under-predicts it in the turbulent one, because
    the model's whole mechanism assumes reversion that never actually
    happens. The tests above distinguish the two cases, at least in
    their effect size if not always in their bare significance; the
    temptation, in practice, is to skip straight to fitting a GARCH
    model without checking which situation is actually present.

## Where this leaves you

The mean of a return series is close to unpredictable. The variance is
not — it clusters, it is measurable, and thirteen chapters of tooling
built for the mean have nothing to say about it. Chapter 25 builds a
model for it, and the idea underneath is considerably simpler than the
size of the literature it launched would suggest.
