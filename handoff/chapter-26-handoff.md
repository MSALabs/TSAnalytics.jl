# Handoff: Chapter 26 — Asymmetry

`docs/src/introduction/26-asymmetry.md`. Target 11–12 pages.

Chapter 25's GARCH treats a 5% fall and a 5% rise as identical shocks.
Markets do not. This chapter fixes that, twice, in two different ways —
and one of the fixes has a naming problem that is a genuine trap.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Two [U] claims — one needs care, one still open

### The omega claim: **did not reproduce, and must be restated**

The recorded claim was *"EGARCH's ω can legitimately be negative
because it models log-variance."* Fitting EGARCH(1,1,1) to a simulated
GARCH series, `n = 1500`:

```
omega     =  0.032578    <- positive
alpha[1]  =  0.177671
gamma[1]  = -0.008089
beta[1]   =  0.957703
```

**Omega came out positive.** The theoretical point is still correct —
EGARCH models log-variance, so there is no positivity constraint on
omega the way there is in GARCH — but "can be negative" is a statement
about what the parameter space permits, not about what fitting a series
produces.

**Write it as the former, not the latter.** If the chapter wants to
show a negative omega it must find a series that produces one, and if
none does, the honest version is: *EGARCH imposes no positivity
constraint, which is a structural difference from GARCH whether or not
a given fit exercises it.*

### The alpha/gamma naming claim: **still [U]**

The recorded claim is that EGARCH's `α` and `γ` are not consistently
named across packages — one implementation's `γ` is another's `α`. This
could not be verified: `arch` was checked and R's `rugarch` was not,
because CRAN has been unreachable throughout this project.

**Do not write it as established.** Either verify it during drafting if
CRAN becomes reachable, or state it as a caution that the reader should
check their own package's parameterisation — which is good advice
regardless of whether the specific claim holds.

---

## 2. The chapter, beat by beat

### Beat 1 — Falls and rises are not the same (about 1.5 pages)

**Chart 1 — a real return series with the largest falls and the largest
rises marked, and the realised volatility over the following twenty
days for each.**

*Reading:* volatility after large falls is systematically higher than
volatility after large rises of the same size. Report the actual
numbers from the run.

This is the leverage effect, named for an explanation that is probably
not the main mechanism — a fall raises a firm's debt-to-equity ratio and
so raises its equity risk. The name stuck; the explanation is disputed;
the pattern is robust across markets and decades.

**Chart 2 — GARCH(1,1)'s response to a large fall and a large rise,
overlaid.**

*Reading:* identical. The model squares the shock, and squaring
destroys the sign before it reaches the variance equation. Chapter 25's
model cannot represent asymmetry no matter how it is fitted, because the
information is discarded at the first step.

### Beat 2 — Add a term for bad news (about 2.5 pages)

**Chart 3 — the news impact curve for GARCH and for GJR-GARCH, on one
set of axes.**

Plot next-period variance against this-period shock, across a range of
positive and negative shocks.

*Reading, and this is the chapter's best single chart:* GARCH gives a
symmetric parabola. GJR gives a parabola with a kink at zero — steeper
on the left. The picture *is* the model, and it makes the whole idea
legible in a way that the equation does not.

Explain the mechanism: GJR adds one term that switches on only when the
shock is negative. One extra parameter, `γ`, and if `γ > 0` bad news
raises variance more than good news.

**Chart 4 — GJR fitted to a real series, with `γ` and its standard
error reported.**

*Reading:* report what actually comes out. On equity index data `γ` is
usually positive and significant; on currency data it is often not,
because currency moves lack a natural asymmetry — a fall in one currency
is a rise in another.

That contrast is worth showing if a suitable pair exists in the bundle.

### Beat 3 — Model the logarithm instead (about 2.5 pages)

**Chart 5 — the news impact curve for EGARCH, added to chart 3's axes.**

*Reading:* a different shape again — asymmetric, and smooth rather than
kinked.

Explain the structural difference: EGARCH models `log(variance)`, so
the variance is positive automatically whatever the parameters do. GARCH
and GJR both need explicit non-negativity constraints and an optimiser
that respects them; EGARCH does not.

**That is the real argument for EGARCH and it is a computational one as
much as a statistical one.** State it plainly, and note the
consequence from section 1: because there is no positivity constraint,
`omega` is unconstrained in sign — whether a given fit produces a
negative one is a separate matter.

**Chart 6 — EGARCH fitted to the same series as chart 4, parameters
reported.**

*Reading:* use the verified numbers or your own. And here the naming
caution belongs: EGARCH parameterisations differ between packages, so
`α` and `γ` may not mean what a reader's previous software called them.
State this package's parameterisation explicitly and advise checking
any other.

### Beat 4 — Which one (about 2 pages)

**Chart 7 — GARCH, GJR and EGARCH fitted to the same series, with
their conditional variances overlaid and their information criteria
reported.**

*Reading:* the three variance paths are similar in calm periods and
diverge after large moves, which is exactly where they were designed to
differ. Report the AIC/BIC and say which wins on this series.

Do not generalise from one series. If GJR wins here and EGARCH wins on
another, that is the honest state of the literature — neither dominates
and the choice is usually made by convention or by which one converges.

**Chart 8 — standardised residuals from all three, with ARCH-LM
p-values.**

*Reading:* chapter 25's diagnostic applied to all three. If all three
pass, the choice between them is not a specification question and the
reader should stop agonising over it.

### Beat 5 — What asymmetry costs (about 1.5 pages)

**Chart 9 — GJR and GARCH forecasts of variance following a large fall,
side by side.**

*Reading:* GJR predicts substantially higher variance after the fall.
For anyone setting a risk limit that difference is the entire point of
the model, and it is largest exactly when it matters most.

Note the honest counterweight: the extra parameter has to be estimated,
and on a short series the asymmetry term may be poorly identified. The
model is not free.

### Beat 6 — Where this leaves you (half a page)

You can model variance that moves and responds asymmetrically to news.

Everything so far has been in-sample description. Nothing has forecast
a variance forward more than implicitly — and one of these three models
turns out to be structurally unable to do it analytically.

Chapter 27.

No recap.

---

## 3. The `india` box

Placed in beat 2, after chart 4.

The leverage effect is well documented for Indian equity indices and
the asymmetry parameter is typically positive and significant for the
Nifty and Sensex — the same pattern as developed markets, which is
worth stating because it is not automatic.

The rupee is a more interesting case. A managed float with periodic
central bank intervention does not produce the clean asymmetry that a
freely floating currency or an equity index does, and an asymmetry term
fitted to it may be capturing intervention rather than leverage.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Tsay ch. 3** | Both models, the news impact curve, and the practical comparison. Tsay is the reference for this chapter | The full variant zoo |
| **Nelson (1991), Glosten, Jagannathan & Runkle (1993)** | Cite both properly | Derivations |
| **Engle & Ng (1993)** | The news impact curve itself, which is the chapter's central teaching device | The full test battery |
| **Shumway & Stoffer** | Fitted comparisons on real data | — |
| **Hamilton** | Little beyond the GARCH foundation | — |
| **Montgomery, Jennings & Kulahci** | Nothing; asymmetric volatility is a financial phenomenon | — |

**On examples:** `nyse`, `sp500w`, `djia`, `gafa_stock` bundled. If a
currency series is available it makes the beat 2 contrast much
stronger; check the bundle.

---

## 5. Voice

**The news impact curve carries the chapter.** Three curves on one set
of axes explain three models better than three sets of equations. Build
it carefully.

**Be honest that the model choice is not settled.** GJR and EGARCH both
work, neither dominates, and a great deal of published comparison is
noise. A reader who expects a verdict should be told there isn't one.

**Handle the naming caution without overclaiming.** The specific
cross-package claim is unverified. The general advice — check your
package's parameterisation — is sound and costs nothing.

Avoid, beyond earlier lists:

- Explaining the leverage effect as though the mechanism were settled.
  It is not.
- Listing further variants. TGARCH, APARCH, FIGARCH and the rest exist
  and belong nowhere near this chapter.

---

## 6. Checklist

- [ ] Nine charts through `@example ch26`
- [ ] Post-fall versus post-rise volatility measured on real data
- [ ] GARCH shown being structurally unable to represent asymmetry
- [ ] **News impact curves for all three models on one set of axes**
- [ ] `γ` reported with its standard error on a real fit
- [ ] EGARCH's no-positivity-constraint argument made structurally,
      **not** as a claim that omega will be negative
- [ ] Naming caution stated as a caution, not as a verified finding
- [ ] Three models compared with information criteria, no
      over-generalisation from one series
- [ ] Standardised-residual diagnostics for all three
- [ ] `india` box on Nifty asymmetry and the managed-float caveat
- [ ] Ends by opening chapter 27, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. **Restate the omega point correctly.** It did not reproduce as
   recorded. Either find a series that produces a negative omega and
   show it, or make the structural argument without the empirical
   claim.
2. **Retry CRAN.** If `rugarch` installs, verify the EGARCH naming
   claim and upgrade it. If not, keep it as a caution.
3. Check the bundle for a currency series to strengthen beat 2's
   contrast.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
