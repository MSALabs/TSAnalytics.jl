# Handoff: Chapter 18 — Fitting ARMA

`docs/src/introduction/18-fitting-arma.md`. Target 12–13 pages.

Chapter 17 showed what these processes look like when you already know
the answer. This chapter is about getting the answer from data — and it
carries a disagreement box that is more unsettling than most, because
four respectable methods give four different answers to the same
question.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Two [U] claims, re-verified — one confirmed and enlarged, one still open

`all-chapters-handoff.md` tagged both of this chapter's disagreements
**[U]** and said to re-verify. Done.

### The standard-error claim: confirmed, and larger than recorded

The recorded claim was "R defaults to Hessian standard errors, Python to
OPG". Verified on a simulated ARMA(1,1), `n = 200`:

**The coefficients agree to five decimals:**

```
R      ar1 = 0.621619   ma1 = 0.199567
Python ar1 = 0.621628   ma1 = 0.199555
```

**The standard errors do not:**

| method | SE(ar1) | SE(ma1) |
|---|---|---|
| Python `opg` (its default) | 0.073970 | 0.099950 |
| Python `oim` (observed information) | 0.075563 | 0.095121 |
| Python `robust` (sandwich) | 0.077804 | 0.090979 |
| R `arima` | 0.074381 | 0.092567 |

**R matches none of Python's three options.** The recorded claim
implied a two-way difference; it is really a four-way one, and the
practical consequence is concrete. For the MA coefficient of 0.199555,
the implied t-statistics are:

```
opg    : 2.00   <- right at the 5% boundary
oim    : 2.10
R      : 2.16
robust : 2.19
```

So the same fitted model, on the same data, is marginally significant
or comfortably significant depending only on which covariance estimator
the software happened to default to. That is the box.

### The transformed-Hessian claim: still unverified

The second recorded claim — that computing the Hessian in the
*transformed* (Monahan) parameter space produces plausible but wrong
standard errors, a bug this project reportedly caught — **could not be
reproduced this session**, and I did not attempt it. It remains **[U]**.

**Do not write it as established.** Either reproduce it during drafting
and include it, or omit it. A book claiming a bug it has not
demonstrated is exactly the failure the STL episode taught.

---

## 2. The chapter, beat by beat

### Beat 1 — The parameters are unknown (about 1 page)

**Chart 1 — a real series and three candidate AR(1) fits at φ = 0.3,
0.6 and 0.9, overlaid.**

*Reading:* one of these tracks the data better than the others, and the
eye can rank them roughly. Ranking is not estimating. What is needed is
a rule that turns "better" into a number, so that a computer can search
rather than a human squint.

### Beat 2 — Least squares, and why it is not enough (about 2 pages)

The obvious attempt.

**Chart 2 — the conditional sum of squares surface for an AR(1),
plotted against φ.**

*Reading:* a clean curve with a clear minimum. For a pure AR model this
works and is genuinely how it used to be done — regress the series on
its own lag.

**Chart 3 — the same attempt on an MA(1), and why it fails.**

*Reading:* the MA model's residuals depend on previous residuals, which
depend on previous residuals, all the way back to observations that do
not exist. Least squares needs starting values it does not have. The
conditional approach assumes the pre-sample shocks were zero, which is
wrong and matters most for short series — precisely where you can least
afford it.

### Beat 3 — The likelihood (about 2 pages)

**Chart 4 — the likelihood surface for an ARMA(1,1), as a contour plot
over (φ, θ).**

*Reading:* a maximum, and a ridge running through the surface. The
ridge is worth pointing out — it means some combinations of φ and θ are
nearly equally good, which is why standard errors can be large even
when the fit is excellent, and why an optimiser can wander.

Explain the prediction-error decomposition briefly: the likelihood is
built from one-step-ahead forecast errors and their variances, which
means it is computed by *running the model forward through the data*.
Do not develop it further — chapter 30 reveals that this machinery is
the Kalman filter, and that reveal should not be spoiled here.

### Beat 4 — Staying inside the region (about 2 pages)

Chapter 17 drew the stationarity triangle. The optimiser must respect
it.

**Chart 5 — an unconstrained optimiser's path wandering outside the
stationarity region, beside a constrained one staying inside.**

*Reading:* the naive approach is to reject any step that leaves the
region, which produces a search that keeps hitting a wall. The Monahan
transform instead re-parameterises so that the *entire* unconstrained
space maps into the valid region — the optimiser can go anywhere, and
every point it visits is legal.

**`julia` box here.** The transform must be differentiable for
automatic differentiation to work through it, and this project found
that writing it type-generically rather than annotating `Float64` was
what made `ForwardDiff` work. Half a page, concrete, no advocacy.

### Beat 5 — Four answers to one question (about 3 pages)

**Chart 6 — the fitted coefficients from R and Python plotted with
their confidence intervals, using all four standard-error methods.**

*Reading:* the point estimates sit on top of each other. The intervals
do not. Use the verified table from section 1.

Then the `disagreement` box, told properly:

The coefficient estimates agree to five decimal places, which is
reassuring and slightly misleading, because it suggests the two
implementations agree. They do not agree about how *certain* those
estimates are.

Three estimators are in play. The observed information matrix is the
curvature of the likelihood at its peak — a sharp peak means a precise
estimate. The outer product of gradients estimates the same quantity
differently and is cheaper. The sandwich estimator combines both and
stays valid under some kinds of misspecification. All three are correct
under their own assumptions; they differ because the assumptions
differ.

Then the consequence, stated flatly: on this fit the MA coefficient's
t-statistic ranges from 2.00 to 2.19 depending only on the default.
Under `opg` it sits exactly on the significance boundary.

State what this package does and why, and note that R's answer matches
none of Python's three, which suggests a fourth variation in the
numerical details rather than a different estimator family.

### Beat 6 — Where this leaves you (half a page)

You can fit an ARMA model to a stationary series and you know how much
to trust the standard errors — which is less than the software's
confident output implies.

Almost no interesting series is stationary. Chapter 19 removes that
restriction.

No recap.

---

## 3. The `india` box

Placed in beat 5, after the disagreement box.

Indian quarterly macroeconomic series often run to only sixty or eighty
observations. All three covariance estimators are justified
asymptotically, and at `n = 80` "asymptotically" is doing real work. The
spread between them widens as the sample shrinks, so the significance
of a coefficient in a short Indian series can genuinely depend on
software choice.

The practical response is to report which estimator was used, and
ideally to check whether the conclusion survives the alternatives. That
is a two-line change and almost nobody does it.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Hamilton ch. 5** | Maximum likelihood for ARMA done properly, including why the exact likelihood differs from conditional least squares | The asymptotic distribution theory |
| **Box & Jenkins via Montgomery** | The conditional-sum-of-squares approach and its historical role — this is how it was done before cheap computing | The full workflow |
| **Shumway & Stoffer** | The prediction-error decomposition, and the honest treatment of standard errors | The state-space connection, which chapter 30 needs unspoiled |
| **Monahan (1984)** | The transform itself; cite it properly | — |
| **fpp3 §9.7** | The practitioner's framing — that estimation is something software does and the user's job is checking it | The `fable` syntax |
| **Tsay** | The robust covariance estimator's motivation, which is a financial-econometrics concern before it is anyone else's | Volatility |

**On examples:** `GNP23`, `rec`, `varve` bundled. The ARMA(1,1) for
the disagreement box is simulated with a fixed seed and should be
reproducible from the text.

---

## 5. Voice

**Beat 5 must not read as a criticism of either package.** All three
estimators are legitimate and the disagreement is a genuine statistical
subtlety, not a bug. What is criticisable is that the choice is a
silent default nobody reads, and that is the point to press.

**Do not spoil chapter 30.** The likelihood is computed by a recursion
that is exactly the Kalman filter, and the reveal in Part VI depends on
the reader not having been told. Describe the mechanism functionally
and leave its name alone.

Avoid, beyond earlier lists:

- Deriving the likelihood. Chart 4 does the work.
- Presenting one covariance estimator as correct.
- The phrase "under regularity conditions" without saying which.

---

## 6. Checklist

- [ ] Six charts through `@example ch18`
- [ ] Likelihood surface shown as a contour, with the ridge noted
- [ ] Monahan transform motivated by chart 5's wandering optimiser
- [ ] `julia` box on differentiability and type-generic code
- [ ] `disagreement` box uses the verified four-way SE table
- [ ] The t-statistic consequence (2.00 to 2.19) stated explicitly
- [ ] **The transformed-Hessian claim either reproduced or omitted** —
      not asserted
- [ ] Kalman filter not named
- [ ] `india` box on short samples and estimator spread
- [ ] Ends by opening chapter 19, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the four-way standard-error comparison during drafting, with
   a stated seed, so the table is reproducible from the text.
2. **Decide the transformed-Hessian claim.** Attempt to reproduce it;
   if it does not reproduce, drop it and say nothing. It is currently
   the only unverified claim in the chapter.
3. Confirm which covariance estimator this package uses by default
   before writing that sentence.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
