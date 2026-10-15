# The Components You Cannot See

```@example ch52
using TSAnalytics, Plots, DelimitedFiles, Statistics
nile = readdlm(TSAnalytics.NILE, ',', skipstart = 1)[:, 2]
years = 1871:1970
println("mean flow 1871-1898: ", round(mean(nile[1:28]), digits = 1),
        "   1899-1970: ", round(mean(nile[29:end]), digits = 1))
```

```@example ch52
plot(years, nile; color = :black, label = "Nile flow at Aswan", size = (700, 300),
     ylabel = "10⁸ m³", legend = :topright)
vline!([1899]; color = :firebrick, linestyle = :dash, label = "1899")
```

Part III took a series apart with procedures: moving averages in
[Chapter 14](14-classical-decomposition.md), local regression in
[Chapter 15](15-stl.md). They return a trend and a seasonal, but not how
sure they are about either, because nothing in them is a model of the
data. This chapter takes a series apart with a model — one in which the
level, the slope and the seasonal are unobserved random processes, each
with a variance to estimate — and gets back what a procedure cannot: every
component with a standard error, a likelihood to compare specifications
by, and forecasts that follow from the same equations.

The Nile is the canonical example (Durbin & Koopman 2012, ch. 2). Its
mean fell by about a fifth after 1898, when the first Aswan dam was
built. A good decomposition should find the drop without being told when it
happened.

These models are also called *structural time series models* (Harvey
1989; Tsay 2010, §11.3.5). That is unrelated to the *structural VAR* of
[Chapter 51](51-shocks-with-names.md); the clash of names predates both.

!!! note "Notation"
    This chapter writes the state-space form as Tsay (2010, Eqs.
    11.26–11.27, printed pp. 576–577, PDF 603–604) and Durbin & Koopman
    do, `y_t = Z α_t + ε_t`, `α_{t+1} = T α_t + R η_t`, with variances `H`
    and `Q` — the field names of the package's [`TimeVaryingSSM`](@ref).
    Hyndman & Athanasopoulos (fpp, Ch. 9) write the same model with `G`,
    `f` and `W`; Shumway & Stoffer with `Φ`, `A` and `Q`, `R`; Hamilton
    with `F`, `H′`. The models are the same.

## A level that wanders

The **local level model** says the series is a level plus noise, and the
level is a random walk:

```math
y_t = \mu_t + \varepsilon_t, \qquad \mu_{t+1} = \mu_t + \eta_t, \qquad
\varepsilon_t \sim N(0, \sigma^2_\varepsilon), \quad \eta_t \sim N(0, \sigma^2_\eta)
```

Two variances, and they pull in opposite directions. A large
`σ²_ε` says the year-to-year scatter is mostly noise around a steady
level; a large `σ²_η` says the level itself moves. Only their ratio, the
**signal-to-noise ratio** `q = σ²_η / σ²_ε`, decides how the model reads a
surprise.

```@example ch52
m = uc_local_level(nile)
q = m.params[2] / m.params[1]
println("σ²_ε = ", round(m.params[1], digits = 1), "   σ²_η = ", round(m.params[2], digits = 1),
        "   q = ", round(q, digits = 4))
```

About a tenth: in a typical year, a tenth as much of the change is a real
move in the level as is passing noise.

## Filtering and smoothing

There are two ways to estimate the level in 1899, and the difference
between them is the most important idea in this Part. The **filter** uses
the data up to 1899 — it is what someone standing in 1899 could know. The
**smoother** uses the whole century. The Kalman filter of
[Chapter 34](34-the-kalman-filter.md) computes the first by a recursion
that alternates two steps, and the smoother of [Chapter 35](35-smoothing.md)
runs back over the filter's output:

```@raw html
<svg viewBox="0 0 640 170" width="100%" style="max-width:640px" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="13">
  <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>
  <rect x="10" y="40" width="170" height="60" rx="8" fill="#e8f0f8" stroke="#4682b4"/>
  <text x="95" y="64" text-anchor="middle">predict</text>
  <text x="95" y="84" text-anchor="middle" font-size="12">a(t|t−1), P(t|t−1)</text>
  <rect x="235" y="40" width="170" height="60" rx="8" fill="#fdf2e6" stroke="#d2691e"/>
  <text x="320" y="64" text-anchor="middle">observe y(t)</text>
  <text x="320" y="84" text-anchor="middle" font-size="12">surprise v(t) = y(t) − Z a(t|t−1)</text>
  <rect x="460" y="40" width="170" height="60" rx="8" fill="#e8f0f8" stroke="#4682b4"/>
  <text x="545" y="64" text-anchor="middle">update</text>
  <text x="545" y="84" text-anchor="middle" font-size="12">a(t|t) = a(t|t−1) + K v(t)</text>
  <line x1="180" y1="70" x2="232" y2="70" stroke="#555" marker-end="url(#ah)"/>
  <line x1="405" y1="70" x2="457" y2="70" stroke="#555" marker-end="url(#ah)"/>
  <path d="M545,100 C545,150 95,150 95,103" fill="none" stroke="#555" marker-end="url(#ah)"/>
  <text x="320" y="160" text-anchor="middle" font-size="12" fill="#555">t → t + 1: a(t+1|t) = T a(t|t)</text>
  <text x="320" y="22" text-anchor="middle" font-size="12" fill="#555">the gain K is large when the level is uncertain relative to the noise</text>
</svg>
```

The update moves the estimate a fraction `K` of the way towards each new
observation. For the local level, `K` settles to a constant fixed by `q`
alone, `K = (−q + √(q² + 4q)) / 2`. Around the dam:

```@example ch52
d = uc_decomposition(m)
println("1897-1903, filtered (one step ahead): ", round.(Int, m.fitted[27:33]))
println("1897-1903, smoothed:                  ", round.(Int, d.level[27:33]))
```

```@example ch52
plot(years, nile; color = :gray, alpha = 0.5, label = "data", size = (700, 340))
plot!(years[2:end], m.fitted[2:end]; color = :darkorange, linewidth = 2, label = "filtered (one step ahead)")
plot!(years, d.level; color = :steelblue, linewidth = 2, label = "smoothed")
vline!([1899]; color = :firebrick, linestyle = :dash, label = "")
xlims!(1880, 1925)
```

The filter learns about the dam only from the low years themselves, so it
lags: it is still above 1000 in 1900 and takes until about 1903 to reach
the new level. The smoother knows the future, and starts coming down
before 1899 — it spreads the break over a few years on both sides, because
a random-walk level has no way to represent a step except as a run of
moves. That trade is the honest one. The smoother is the better estimate
of what *happened*; only the filter is available to someone *forecasting*.
Cowpertwait & Metcalfe (2009, Figs. 12.1–12.2) draw the same contrast across
a level shift in a simulated series; it is the clearest picture of the
distinction there is.

## The model behind simple exponential smoothing

Simple exponential smoothing ([Chapter 24](24-weighting-the-recent-past.md))
updates a level by a fixed fraction `α` of each forecast error — the
filter's update with `K = α`. The local level is the model for which that
recursion is optimal, so the two estimate the same quantity two ways:

```@example ch52
K = (-q + sqrt(q^2 + 4q)) / 2
e = fit_ets(nile; error = :add, trend = :none, seasonal = :none)
println("gain implied by the UC variances: ", round(K, digits = 3),
        "    α estimated by ETS(A,N,N): ", round(e.alpha, digits = 3))
```

Close but not equal, and for a reason: ETS estimates `α` and the starting
level by maximising the likelihood of its own error recursion, while the
UC model estimates the two variances and lets `K` follow. Montgomery,
Jennings & Kulahci (2008, §4.8, printed pp. 217–220, PDF 230–233) give the
same identity in its classical form, simple exponential smoothing as the
optimal forecast of an IMA(1,1) with `θ = 1 − λ`. What the UC form adds
is the decomposition: `α` alone cannot tell you how much of a year's
change was the level.

## Adding a seasonal: Johnson & Johnson

A seasonal component is a set of `s` effects that sum to a zero-mean
disturbance, so it can evolve slowly from year to year:

```math
\gamma_{t+1} = -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t, \qquad \omega_t \sim N(0, \sigma^2_\omega)
```

Tsay (2010, Example 11.3, printed p. 606, PDF 633) fits a local level
and a quarterly seasonal to the log of Johnson & Johnson's quarterly
earnings per share, in S-PLUS with SsfPack:

```@example ch52
y = log.(dataset("jj").value)
s = fit_uc(y; level = :stochastic, seasonal = 4)
sd = sqrt.(s.params)
println("σ_η (level)    ", round(sd[2], sigdigits = 7), "    Tsay: 0.07269655")
println("σ_ω (seasonal) ", round(sd[3], sigdigits = 7), "    Tsay: 0.02931691")
println("σ_ε (irregular) ", round(sd[1], sigdigits = 3), "    Tsay: 2.044516e-06")
```

The two standard deviations that matter agree with a different
implementation, in a different language, sixteen years older, to about
`1e-7`. The irregular does not, and should not be expected to: both are
effectively zero, and a variance at the edge of its parameter space is
estimated by wherever an optimiser happens to stop
([Chapter 53](53-when-a-component-stops-moving.md) is about exactly this).

```@example ch52
plot(state_decomposition(s); size = (700, 520))
```

The redraw of Tsay's Figure 11.6, from our own fit: the level, with a 95%
band for the smoothed component, and a seasonal pattern that grows in
amplitude through the 1960s and 70s. That growth is the seasonal variance
at work — a fixed seasonal could not do it.

### Seasonal adjustment from a model

Subtracting the smoothed seasonal gives a seasonally adjusted series, and
because the seasonal has a standard error so does the adjustment:

```@example ch52
dj = uc_decomposition(s)
adj = y .- dj.seasonal
st = stl_decompose(y, 4)
println("largest gap between the UC and STL adjustments: ",
        round(maximum(abs, adj .- (y .- st.seasonal)), digits = 4), " log points")
```

```@example ch52
dts = dataset("jj").date
plot(dts, y; color = :gray, alpha = 0.6, label = "log EPS", size = (700, 320), legend = :topleft)
plot!(dts, adj; color = :steelblue, linewidth = 2, label = "adjusted: UC smoother")
plot!(dts, y .- st.seasonal; color = :darkorange, linestyle = :dash, label = "adjusted: STL")
```

Two adjustments that agree to within about five hundredths of a log
point at worst. They should: both are reasonable. The point of the model-based one is not a different
answer but the band it can carry and the likelihood behind it — the
official agencies' X-13 of [Part IX](42-official-seasonal-adjustment.md)
offers neither.

## A cycle: the lynx

A **stochastic cycle** is a damped rotation with its own frequency,
damping and disturbance — a component that oscillates without being
locked to the calendar. The lynx trappings of
[Chapter 56](56-a-different-model-on-each-side-of-a-line.md) have the
best-known cycle in ecology:

```@example ch52
lx = log10.(readdlm(TSAnalytics.LYNX, ',', skipstart = 1)[:, 2])
c = fit_uc(lx; level = :deterministic, cycle = :stochastic, damped_cycle = true)
λ = c.params[findfirst(==("frequency.cycle"), c.layout.names)]
ρ = c.params[findfirst(==("damping.cycle"), c.layout.names)]
println("estimated period: ", round(2π / λ, digits = 2), " years    damping: ", round(ρ, digits = 3))
```

```@example ch52
dc = uc_decomposition(c)
plot(1821:1934, dc.cycle; ribbon = 1.96 .* dc.cycle_se, label = "cycle, 95% band",
     size = (700, 300), legend = :topright)
```

A period of about ten and three-quarter years, estimated, not assumed —
the frequency is a parameter like any other, which is why
[`fit_uc`](@ref) tries several starting frequencies (the likelihood has a
local optimum near every harmonic). [Chapter 56](56-a-different-model-on-each-side-of-a-line.md)
reads the same cycle a different way, as a series whose dynamics change
with its own level.

!!! disagreement "When Implementations Disagree"
    R's `StructTS` reports a `loglik` that cannot be a log-likelihood.
    The Gaussian state-space log-likelihood (fpp, printed p. 99, PDF 99)
    has `−(T/2) log 2π` — a **negative** normalising term — but
    `StructTS`'s two reported figures differ by exactly
    `n·log 2π + ½·n_obs·log 2π`, a positive multiple, and the value it
    reports for the Nile is `10.663` from the true likelihood at its own
    parameters. It also starts the level at the first observation, where
    the standard convention (fpp, printed p. 98) is a large prior variance,
    and clips negative variance estimates to zero. `statsmodels`'
    `UnobservedComponents` gets all three right and agrees with this
    package to `1e-10`. Compare variances with `StructTS`, never
    likelihoods.

!!! julia "Under the Hood"
    [`fit_uc`](@ref) runs its starting points on threads and keeps the
    best. Writing that exposed a classic Julia trap: a closure that
    assigns a variable the enclosing function *also* assigns shares one
    boxed variable with it, and with threads, every task wrote into the
    same box — six identical fits returned four different answers. The
    cure is one word, `local`, and the test suite now refits the same
    model on four threads and requires bit-identical results. Data races
    in Julia are not exotic; they are a scoping rule meeting
    `Threads.@threads`.

## Summary

- An unobserved-components model decomposes a series into components that
  are random processes, each with a variance; the filter estimates them in
  real time, the smoother with hindsight, and both carry standard errors.
- Only variance *ratios* matter to the shape of the decomposition; the
  local level's ratio fixes the gain of simple exponential smoothing.
- Seasonals and cycles can evolve, and how much they evolve is estimated.
- The same machinery seasonally adjusts with a band, and agrees with STL
  where STL is right.

## See also

- [Manual: Unobserved Components](../manual/17-unobserved-components.md) — the task-oriented version
- [Chapter 53](53-when-a-component-stops-moving.md) — what a variance of zero means
- [Chapter 33](33-the-state-space-form.md) — the state-space form these models are written in
