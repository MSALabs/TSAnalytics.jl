# Handoff: Stage 10.3 — Impulse responses and forecast error variance decomposition

**Status: implemented ✅ (2026-10-13).**

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first** — it carries the stage-wide
conventions (§5, including the running list of reference defects at
§5.10), the architecture decisions (§6), the reference-environment recipe
(§3) and the performance and test policy (§7, §8). This document cites
them rather than repeating them.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document** — re-verify any package-internal
claim before building on it.

---

## 1. Where this fits

- **Depends on:** 10.1 (`VARModel`, and specifically its `vcov_full`
  field and `companion`), plus 10.0's companion matrix. Nothing else.
- **Feeds into:** 10.7 (SVAR reuses the whole MA-representation and
  standard-error machinery with a different `P`), and 10.6 (a VECM's MA
  representation).
- **Build it second**, right after 10.1, because `[V]` both references
  agree to machine precision on every point estimate (§3) — so the
  implementation is checked against consensus rather than adjudicated.
- **Reference size:** `[V]` the point estimates are tiny —
  statsmodels' `ma_rep` is **14 lines**, `orth_ma_rep` **18**, `fevd`
  **17**. The uncertainty machinery is where the volume is, and it
  splits cleanly (§5, §6).

---

## 2. Scope

### In

- MA representation: `Phi_i` (non-orthogonalized) and `Psi_i = Phi_i P`
  (orthogonalized), `i = 0 … h`.
- Impulse responses: orthogonalized and non-orthogonalized, plain and
  cumulative.
- Long-run (total) effects, `(I - sum A_j)^-1`, and their standard errors.
- FEVD.
- **Analytic asymptotic standard errors** for all of the above —
  independently verified two ways (§5).
- **Bootstrap confidence bands**, residual resampling, seeded.
- Plot recipes for IRF and FEVD.

### Out (stated rather than hidden)

- **Sims-Zha error bands.** `[V]` statsmodels implements three variants
  (`err_band_sz1/2/3`, 94 + 97 + 117 lines, plus
  `_eigval_decomp_SZ` 38); `vars` has none. They are a separate, large,
  optional feature. Note them as unimplemented; do not half-build them.
- Structural IRFs (SVAR identification) — 10.7, which reuses everything
  here with a non-Cholesky `P`.
- Generalized (Pesaran-Shin) impulse responses. Neither reference ships
  them; a real later differentiation opportunity.
- Historical decompositions, local projections (`statsmodels` has
  `local_proj.py`, 469 lines; out of roadmap scope).

---

## 3. Point estimates: both references agree to machine precision

`[V]` Across six fixtures, four IRF variants, both MA representations
and the FEVD — **every single quantity agrees**. Worst case is `econ5`,
whose data are of order 10⁴:

| Fixture | ortho IRF | non-ortho IRF | ortho cumulative | non-ortho cumulative | Psi | Phi | FEVD |
|---|---|---|---|---|---|---|---|
| `var2_p1` p=1 h=6 | 3.3e-16 | 2.5e-16 | 1.1e-15 | 6.7e-16 | 3.3e-16 | 2.5e-16 | 5.6e-16 |
| `var2_p1` p=2 h=8 | 1.0e-15 | 1.4e-15 | 2.9e-15 | 4.7e-15 | 1.0e-15 | 1.4e-15 | 3.3e-16 |
| `var3_p2` p=2 h=6 | 3.3e-16 | 4.4e-16 | 1.6e-15 | 1.3e-15 | 3.3e-16 | 4.4e-16 | 4.4e-16 |
| `uschange` p=2 h=6 | 3.9e-14 | 2.4e-14 | 7.3e-14 | 2.5e-14 | 3.9e-14 | 2.4e-14 | 6.3e-15 |
| `econ5` p=2 h=4 | 6.1e-13 | 1.1e-12 | 1.6e-12 | 3.2e-12 | 6.1e-13 | 1.1e-12 | 9.1e-15 |
| `var2_short` p=2 h=4 | 4.4e-16 | 3.9e-16 | 8.9e-16 | 8.9e-16 | 4.4e-16 | 3.9e-16 | 3.3e-16 |

`[V]` FEVD rows sum to 1 to `≤ 4.4e-16` on every fixture.

**So any disagreement found while implementing 10.3 is a bug in the
implementation, not a convention to choose.** That is unusual in this
project and it should shape how the work is done: write the recursion,
compare, and trust the comparison.

### 3.1 The recursion, and a form confirmed against the reference

`[V]` This recursion reproduces statsmodels' `ma_rep` **exactly
(difference 0.0)** on `var2_p1`, p=2, h=4:

```
Phi_0 = I
Phi_i = sum_{j=1}^{min(i,p)} Phi_{i-j} A_j            i = 1 … h
Psi_i = Phi_i P          where P = chol(Sigma)' (lower triangular)
cumulative:  sum_{m=0}^{i} Phi_m     (and likewise for Psi)
long-run:    (I - sum_{j=1}^{p} A_j)^-1              orth: that times P
```

Note the order `Phi_{i-j} A_j`, not `A_j Phi_{i-j}`. With `A_j[i,m]` the
coefficient of equation `i` on variable `m` at lag `j` (10.1's layout),
that ordering is the one that matches both references.

### 3.2 Which Σ the orthogonalization uses

`[V]` **The df-adjusted Σ, not the ML one.** Verified three ways on all
six fixtures: R's `t(chol(Sigma_df))` equals Python's
`cholesky(sigma_u)` to `≤ 6.7e-14`, and both equal the orthogonalized
IRF at `h = 0` exactly. On `var2_p1` p=2:

```
P = [1.005858110588  0.0
     0.295542029799  0.731322772863]
```

and `Psi_0` is that matrix. 10.1 stores both Σ conventions (charter
§5.5) — use `sigma_df` here.

---

## 4. Index-layout traps — the one place a silent bug can hide

The two references disagree about *every* array layout while agreeing
about every number. These are mechanical, and one of them is genuinely
dangerous.

| Object | R (`vars`) | Python (statsmodels) |
|---|---|---|
| IRF | `irf$<impulse>` — a **list keyed by impulse**, each `(h+1) × K` with rows = horizon, cols = response | `irfs[horizon, response, impulse]` — one `(h+1, K, K)` array |
| MA rep | `Psi`/`Phi` — `(K, K, h+1)`, **horizon LAST** | `orth_ma_rep`/`ma_rep` — `(h+1, K, K)`, **horizon FIRST** |
| FEVD | `fevd$<variable>` — `h × K`, rows = horizon **1…h** | `decomp[variable, horizon, shock]` — `(K, h, K)` |

Two consequences worth their own tests:

**(a) IRF and FEVD do not share a horizon axis.** `[V]` For the same
call, the IRF has `h+1` horizons starting at 0 and the FEVD has `h`
starting at 1. On `var2_p1` p=2 h=8, `irf_ortho$y1` is 9×2 and
`fevd$y1` is 8×2. An implementation that returns both from one object
must either pad or document the offset — **do not silently align them.**

**(b) When `K == h+1`, R's `(K, K, h+1)` shape is square and a transpose
bug is invisible.** `[V]` This was hit for real while preparing this
handoff: `econ5` with K=5, h=4 gives shape `(5,5,5)`, a comparison that
failed to move the horizon axis reported a discrepancy of **34.48** —
which was the comparison's bug, not a reference disagreement. Once the
axis was moved properly it agreed to 6.1e-13.

**Decision for TSAnalytics:** horizon first, as a `Vector` of `K × K`
matrices — `Phi::Vector{Matrix{Float64}}` of length `h+1` — matching
10.1's `A::Vector{Matrix}` and making `Phi[i]` directly indexable.
Response/impulse within each matrix follows the coefficient convention:
`Phi[i][response, impulse]`. A test must use a fixture where
`K == h+1` precisely because that case cannot catch a transpose by
shape (§7.3).

### 4.1 Cholesky ordering dependence

`[V]` The orthogonalized IRF depends on variable order; the
non-orthogonalized one does not. Verified by permuting `var2_p1`'s
columns to `(y2, y1)` and refitting:

```
original,  ortho, impulse y2, h=0..2:   [0.0,            0.731322772863]
                                        [0.051433600383, 0.290719111006]
                                        [-0.037958240328, 0.105660011645]

permuted,  ortho, impulse y2, h=0..2:   [0.788782663023, 0.376876117629]
                                        [0.396667351649, 0.224409352524]
                                        [0.189307846032, 0.045992972734]
```

Completely different — as it must be, since `y2` is now first in the
Cholesky ordering. Meanwhile the **non-orthogonalized** IRF is merely
relabelled: permuted impulse `y1` gives `[0, 1]`, `[0.220514094415,
0.448249746032]` against the original's `[1, 0]`, `[0.448249746032,
0.220514094415]` — the same numbers, columns swapped.

This is the single most important thing to say in the manual: an
orthogonalized IRF is a claim about contemporaneous causal ordering, and
reordering the variables is not a cosmetic change. Both references
leave the user to know this. **TSAnalytics should go further and make
the ordering explicit in the returned object** — store `names` and a
note of which Σ factorization was used, so a plot can be labelled
honestly.

---

## 5. Analytic standard errors: statsmodels is the only reference, and it was verified independently

`[V]` **`vars` has no analytic standard errors at all** — `irf()`
bootstraps (`.boot`, 79 lines in `vars/R/internal.R`). So the charter's
dual-verification rule cannot be satisfied by a second package here.
Two independent checks were run instead, and both pass.

### 5.1 Check one — numerical Jacobian

The analytic covariance is `G_i Sigma_a G_i'` where
`G_i = d vec(Phi_i) / d vec(alpha)`. A central-difference Jacobian of
the §3.1 recursion, with `eps = 1e-6`, reproduces statsmodels'
`IRAnalysis.cov()` on `var2_p1` p=2:

| h | max abs difference | relative |
|---|---|---|
| 0 | 0.000e+00 | 0 |
| 1 | 2.086e-12 | 2.1e-11 |
| 2 | 9.950e-13 | 1.0e-11 |
| 3 | 8.096e-13 | 1.1e-11 |
| 4 | 1.836e-13 | 3.7e-12 |

**And it pins the flattening convention, which is otherwise guesswork.**
`[V]` Two candidate orderings of the coefficient array were tried
against statsmodels' `cov_a`:

```
vec([A_1 ... A_p])  -- column-major of the K x Kp stacked matrix:  worst 2.086e-12   <- correct
row-major of (p, K, K):                                            worst 3.855e-02   <- wrong
```

So `Sigma_a` is ordered as Lütkepohl's `alpha = vec([A_1 … A_p])`.
Getting this wrong produces standard errors that look plausible and are
about 40% off. **Assert the ordering directly in a test** (§7.5), do not
infer it.

### 5.2 Check two — parametric bootstrap of the sampling distribution

4000 replications, seed 20261013: simulate from the fitted VAR, refit by
OLS, recompute the IRF, take the standard deviation across
replications. Compared with statsmodels' analytic standard errors:

**Non-orthogonalized**

| h | max relative deviation | bootstrap `[0,0]` | analytic `[0,0]` |
|---|---|---|---|
| 1 | 2.85 % | 0.076630 | 0.077607 |
| 2 | 1.36 % | 0.080080 | 0.080910 |
| 3 | 3.50 % | 0.059408 | 0.061564 |
| 4 | 4.16 % | 0.039492 | 0.041205 |

**Orthogonalized** — the formula statsmodels' own source flags with
`# Lutkepohl typo, cov_sig correct`

| h | max relative deviation | bootstrap `[0,0]` | analytic `[0,0]` |
|---|---|---|---|
| 0 | 1.47 % | 0.051287 | 0.050546 |
| 1 | 2.40 % | 0.074549 | 0.075630 |
| 2 | 1.95 % | 0.071477 | 0.072679 |
| 3 | 3.90 % | 0.058408 | 0.060777 |
| 4 | 4.75 % | 0.041649 | 0.043726 |

At 4000 replications the Monte Carlo standard error of an estimated
standard deviation is about `1/sqrt(2 x 4000) ≈ 1.1%`, so 1.4–4.8%
across the maximum of a 2×2 grid at five horizons is what agreement
looks like. **Both formulas are validated, including the orthogonalized
one whose textbook statement is wrong.**

This is the verification the project's methodology demands when there is
only one software reference — the same approach that settled the ETS and
TBATS interval questions in Stage 9 by simulation rather than by
argument. `scripts/verify_irf_se.py` in this bundle reruns it.

### 5.3 A structural asymmetry at h = 0 that must be preserved

`[V]` `stderr()[0]` is **exactly zero** for the non-orthogonalized IRF —
`Phi_0 = I` is not estimated, so it has no sampling variability. But
`stderr(orth=True)[0]` is **not** zero:

```
ortho stderr at h=0:  [0.050546271896  0.0
                       0.054053169849  0.036750352094]
```

because `Psi_0 = P` depends on the estimated `Sigma`. Note the `[0,1]`
entry is exactly zero — `P` is lower triangular, so that element is
structurally zero and has no variability either.

An implementation that computes one standard-error path and reuses it
for both will get `h = 0` wrong in one direction or the other. Assert
both (§7.5).

### 5.4 Reference values to assert

`[V]` `var2_p1`, p=2, `[h, response, impulse]`:

```
analytic stderr, NON-orthogonalised
h=0  [0.0             0.0           ;  0.0             0.0           ]
h=1  [0.077607372675  0.099412416939;  0.060858832319  0.077958103778]
h=2  [0.080909625540  0.098195872083;  0.066236599672  0.080783562797]
h=3  [0.061563849676  0.075818022245;  0.054373838555  0.067057369364]
h=4  [0.041205385598  0.049102416995;  0.039961923976  0.048301675815]

analytic stderr, ORTHOGONALISED
h=0  [0.050546271896  0.0           ;  0.054053169849  0.036750352094]
h=1  [0.075630456964  0.072748492862;  0.062300740140  0.058854546287]
h=2  [0.072678550507  0.071838205873;  0.060725653549  0.059316975760]
h=3  [0.060777273860  0.055486149812;  0.053167669233  0.049057815869]
h=4  [0.043726290185  0.035939218404;  0.041138659874  0.035324319014]

cumulative-effect stderr (non-orth)
h=0  [0.0             0.0           ;  0.0             0.0           ]
h=1  [0.077607372675  0.099412416939;  0.060858832319  0.077958103778]
h=2  [0.134482146215  0.169475590051;  0.108327813368  0.136645202154]

long-run effects                     long-run effect stderr
[ 1.809214109722  -0.117620060622]   [0.239755388927  0.302223302102]
[ 0.670168128085   1.539882703224]   [0.213950128866  0.269694519570]
```

---

## 6. Bootstrap bands — and why they cannot be cross-verified

`[V]` The two references do not implement the same thing:

- `vars::irf(boot=TRUE, runs=100, ci=0.95, seed=...)` — residual
  resampling, percentile interval.
- statsmodels `errband_mc` (Monte Carlo from the asymptotic
  distribution) and `err_band_sz1/2/3` (Sims-Zha).

Different resampling schemes, different RNGs. **Numerical agreement is
impossible and must not be attempted.** A bootstrap band reproduced from
`vars` would only prove that the same RNG stream was replayed, which is
not a property worth having.

Verify it by **coverage** instead, which is what the band claims:
generate from a known VAR, compute a nominal 95% band, and check the
true IRF falls inside roughly 95% of the time. A tolerance of ±3
percentage points at 1000 replications is reasonable; state the figure
in the test rather than asserting a point value.

`[V]` A `vars` bootstrap run is recorded in `ref_R_irf.json` under
`A_p2_boot` (runs = 100, ci = 0.95, seed = 4321) **for reproducibility
of the reference only** — do not assert against it.

Implementation requirements:

- Resample the fitted residuals **by row** (keeping the
  contemporaneous correlation across equations), rebuild the series
  through the fitted recursion from the real presample values, refit,
  recompute. This is `vars`' scheme.
- `rng::AbstractRNG` keyword, defaulting to `Random.default_rng()`, so
  a seeded run is reproducible. A test asserts that two runs with the
  same seed are bit-identical and that different seeds differ.
- `runs` default 100, matching `vars`, with the docstring noting that
  100 is low for a 95% percentile interval and 1000+ is advisable.
- `Threads.@threads` over replications — they are independent. A test
  asserts the threaded and serial paths agree bit-for-bit **for a fixed
  per-replication RNG stream**; that requires one RNG per replication,
  seeded deterministically from the base seed, not one shared RNG.
  Build it that way from the start.

---

## 7. Test battery

Exact values all `[V]`. The reference set is in `fixtures/ref_R_irf.json`
and `fixtures/ref_py_irf.json`, the independent SE verification in
`fixtures/verify_irf_se.json`. Suggested file: `test/test_irf.jl`.
Target **≥ 250 assertions**.

### 7.1 MA representation

```julia
@testset "ma_rep: recursion vs both references" begin
    Y = read_fixture("var2_p1.csv")
    m = fit_var(Y, 2; deterministic=:const)
    Phi = ma_rep(m, 8)            # Vector of 9 K x K matrices, horizon first

    @test length(Phi) == 9
    @test Phi[1] ≈ I(2)                               # Phi_0 = I exactly
    @test Phi[1] == Matrix{Float64}(I, 2, 2)

    # Phi_1 == A_1
    @test Phi[2] ≈ m.A[1] atol=1e-14
    # Phi_2 == A_1^2 + A_2
    @test Phi[3] ≈ m.A[1]*m.A[1] + m.A[2] atol=1e-13

    # against the reference, non-orthogonalised IRF impulse y1 (column 1)
    @test [Phi[h+1][1,1] for h in 0:4] ≈
          [1.0, 0.448249746032, 0.230668860843, 0.099275105298, 0.03465341684] atol=1e-10
    @test [Phi[h+1][2,1] for h in 0:4] ≈
          [0.0, 0.220514094415, 0.199922849949, 0.129947055977, 0.070834656902] atol=1e-10

    Psi = orth_ma_rep(m, 8)
    P = cholesky(Hermitian(m.sigma_df)).L
    @test Psi[1] ≈ P atol=1e-14                       # Psi_0 = P
    @test Psi[1] ≈ [1.005858110588 0.0
                    0.295542029799 0.731322772863] atol=1e-11
    for h in 1:9
        @test Psi[h] ≈ Phi[h] * P atol=1e-12          # Psi_i = Phi_i P
    end
end
```

### 7.2 Impulse responses — all four variants, six fixtures

```julia
@testset "irf: orthogonalised, var2_p1 p=2" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    r = irf(m, 8)

    # impulse y1 (column 1), responses y1 and y2, h = 0..4
    @test [r.orth[h+1][1,1] for h in 0:4] ≈
          [1.005858110588, 0.471660980117, 0.216680467162,
           0.083193228486, 0.023147996206] atol=1e-10
    @test [r.orth[h+1][2,1] for h in 0:4] ≈
          [0.295542029799, 0.339291246158, 0.243793325356,
           0.141164860983, 0.070285305723] atol=1e-10
    # impulse y2 (column 2) -- note r.orth[1][1,2] == 0 by lower triangularity
    @test [r.orth[h+1][1,2] for h in 0:4] ≈
          [0.0, 0.051433600383, -0.037958240328,
           -0.041233912304, -0.028972654904] atol=1e-10
    @test [r.orth[h+1][2,2] for h in 0:4] ≈
          [0.731322772863, 0.290719111006, 0.105660011645,
           0.025874901889, -0.002386194334] atol=1e-10
    @test r.orth[1][1,2] == 0.0                       # structurally zero
end

@testset "irf: cumulative" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    r = irf(m, 8; cumulative=true)
    @test [r.orth[h+1][1,1] for h in 0:4] ≈
          [1.005858110588, 1.477519090705, 1.694199557867,
           1.777392786353, 1.800540782559] atol=1e-10
    # cumulative is the running sum of the plain IRF
    plain = irf(m, 8)
    for h in 1:9
        @test r.orth[h] ≈ sum(plain.orth[1:h]) atol=1e-11
        @test r.nonorth[h] ≈ sum(plain.nonorth[1:h]) atol=1e-11
    end
end

@testset "irf: every fixture against both references" begin
    for (fx, p, h, tol) in [("var2_p1.csv",     1, 6, 1e-11),
                            ("var2_p1.csv",     2, 8, 1e-11),
                            ("var3_p2.csv",     2, 6, 1e-11),
                            ("uschange.csv",    2, 6, 1e-10),
                            ("econ5.csv",       2, 4, 1e-8),   # values ~1e4
                            ("var2_short.csv",  2, 4, 1e-11)]
        m = fit_var(read_fixture(fx), p; deterministic=:const)
        r = irf(m, h)
        ref = load_ref_irf(fx, p, h)          # from ref_R_irf.json
        for i in 1:(h+1)
            @test r.orth[i]    ≈ ref.orth[i]    atol=tol
            @test r.nonorth[i] ≈ ref.nonorth[i] atol=tol
        end
    end
end
```

### 7.3 Layout: the `K == h+1` transpose trap

```julia
@testset "layout: a transpose bug must not hide at K == h+1" begin
    # econ5 has K=5; at h=4 the MA array is 5 x 5 x 5 and a transposed
    # implementation agrees in SHAPE with a correct one. This fixture is
    # here only to catch that -- it cost a real debugging detour.
    m = fit_var(read_fixture("econ5.csv"), 2; deterministic=:const)
    Phi = ma_rep(m, 4)
    @test length(Phi) == 5 && size(Phi[1]) == (5,5)
    @test Phi[1] ≈ I(5)                     # only true horizon-first
    @test Phi[2] ≈ m.A[1] atol=1e-9
    # and the IRF is NOT symmetric in response/impulse, so transposition shows
    @test Phi[3] != transpose(Phi[3])
    ref = load_ref_irf("econ5.csv", 2, 4)
    for i in 1:5
        @test Phi[i] ≈ ref.Phi[i] atol=1e-8
    end
end

@testset "layout: IRF and FEVD horizon axes differ" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    r = irf(m, 8); f = fevd(m, 8)
    @test length(r.orth) == 9        # horizons 0..8
    @test size(f.decomp, 1) == 8     # horizons 1..8 -- deliberately not 9
    @test f.horizons == 1:8
    @test r.horizons == 0:8
end
```

### 7.4 Cholesky ordering dependence

```julia
@testset "orthogonalised IRF depends on variable order; non-orth does not" begin
    Y = read_fixture("var2_p1.csv")
    m  = fit_var(Y, 2; deterministic=:const)
    mp = fit_var(Y[:, [2,1]], 2; deterministic=:const)
    r, rp = irf(m, 4), irf(mp, 4)

    # orthogonalised: genuinely different
    @test [rp.orth[h+1][1,1] for h in 0:2] ≈
          [0.788782663023, 0.396667351649, 0.189307846032] atol=1e-10
    @test !isapprox(rp.orth[1][1,1], r.orth[1][2,2]; atol=1e-3)

    # non-orthogonalised: the same numbers, relabelled
    for h in 1:5
        @test rp.nonorth[h] ≈ r.nonorth[h][[2,1],[2,1]] atol=1e-12
    end
    # and the model records the ordering it orthogonalised against
    @test m.names  == [:y1, :y2]
    @test mp.names == [:y2, :y1]
end
```

### 7.5 Analytic standard errors

```julia
@testset "irf standard errors: analytic, non-orthogonalised" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    se = irf_stderr(m, 4)                      # Vector of K x K, horizon first

    # h = 0 is EXACTLY zero -- Phi_0 = I is not estimated
    @test all(iszero, se[1])

    @test se[2] ≈ [0.077607372675  0.099412416939
                   0.060858832319  0.077958103778] atol=1e-10
    @test se[3] ≈ [0.080909625540  0.098195872083
                   0.066236599672  0.080783562797] atol=1e-10
    @test se[4] ≈ [0.061563849676  0.075818022245
                   0.054373838555  0.067057369364] atol=1e-10
    @test se[5] ≈ [0.041205385598  0.049102416995
                   0.039961923976  0.048301675815] atol=1e-10
end

@testset "irf standard errors: analytic, ORTHOGONALISED" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    se = irf_stderr(m, 4; orth=true)

    # h = 0 is NOT zero here -- Psi_0 = P depends on the estimated Sigma
    @test !all(iszero, se[1])
    @test se[1] ≈ [0.050546271896  0.0
                   0.054053169849  0.036750352094] atol=1e-10
    @test se[1][1,2] == 0.0                    # P lower triangular: structurally zero
    @test se[2] ≈ [0.075630456964  0.072748492862
                   0.062300740140  0.058854546287] atol=1e-10
    @test se[5] ≈ [0.043726290185  0.035939218404
                   0.041138659874  0.035324319014] atol=1e-10
end

@testset "cumulative and long-run standard errors" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    se = cum_effect_stderr(m, 2)
    @test all(iszero, se[1])
    @test se[2] ≈ [0.077607372675  0.099412416939
                   0.060858832319  0.077958103778] atol=1e-10   # == plain at h=1
    @test se[3] ≈ [0.134482146215  0.169475590051
                   0.108327813368  0.136645202154] atol=1e-10

    lr, lrse = lr_effects(m), lr_effect_stderr(m)
    @test lr ≈ [ 1.809214109722 -0.117620060622
                 0.670168128085  1.539882703224] atol=1e-10
    @test lrse ≈ [0.239755388927  0.302223302102
                  0.213950128866  0.269694519570] atol=1e-10
    # lr_effects must equal (I - sum A_j)^-1
    @test lr ≈ inv(I(2) - m.A[1] - m.A[2]) atol=1e-11
end

@testset "Sigma_a flattening is vec([A_1 ... A_p]), NOT row-major" begin
    # Verified by numerical Jacobian: the wrong flattening is off by 3.9e-2,
    # i.e. plausible-looking and about 40% wrong. Assert the ordering directly.
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    alpha = coef_alpha(m)                 # the vector Sigma_a is ordered against
    @test length(alpha) == 2*2*2
    # vec([A_1 A_2]) column-major: A_1[:,1], A_1[:,2], A_2[:,1], A_2[:,2]
    @test alpha[1:2] ≈ m.A[1][:, 1] atol=1e-14
    @test alpha[3:4] ≈ m.A[1][:, 2] atol=1e-14
    @test alpha[5:6] ≈ m.A[2][:, 1] atol=1e-14
    @test alpha[7:8] ≈ m.A[2][:, 2] atol=1e-14
end
```

### 7.6 Independent verification, reproduced as a test

```julia
@testset "analytic SEs vs a numerical Jacobian (independent of statsmodels)" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    H = 4
    se_an = irf_stderr(m, H)
    # central differences on the MA recursion, eps = 1e-6
    se_num = numerical_irf_stderr(m, H; eps=1e-6)     # test helper
    for h in 1:(H+1)
        @test se_num[h] ≈ se_an[h] atol=1e-8
    end
end

@testset "analytic SEs vs a parametric bootstrap (slow; tagged)" begin
    # 4000 reps in Python gave 1.4-4.8% relative agreement at h=0..4.
    # Run 2000 here and allow 8% -- this is a sanity net, not a precision test.
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    se_bs = parametric_bootstrap_irf_sd(m, 4; reps=2000, rng=MersenneTwister(20261013))
    se_an = irf_stderr(m, 4)
    for h in 2:5
        @test maximum(abs.(se_bs[h] .- se_an[h]) ./ se_an[h]) < 0.08
    end
end
```

Tag the bootstrap testset so it can be skipped in a fast run — the
project has no precedent for this, so follow whatever `test/runtests.jl`
already does for the `*_bulk.jl` files.

### 7.7 FEVD

```julia
@testset "fevd: var2_p1 p=2 h=8" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    f = fevd(m, 8)

    # decomp[variable, horizon, shock]; horizons are 1..8
    @test f.decomp[1, 1:5, 1] ≈ [1.0, 0.997861184632, 0.996820665889,
                                 0.995527783322, 0.994884459095] atol=1e-11
    @test f.decomp[1, 1:5, 2] ≈ [0.0, 0.002138815368, 0.003179334111,
                                 0.004472216678, 0.005115540905] atol=1e-11
    @test f.decomp[2, 1:5, 1] ≈ [0.140385996958, 0.246361806681, 0.293472518149,
                                 0.308678100405, 0.312396566803] atol=1e-11
    @test f.decomp[2, 1:5, 2] ≈ [0.859614003042, 0.753638193319, 0.706527481851,
                                 0.691321899595, 0.687603433197] atol=1e-11

    # every row is a proportion decomposition
    for i in 1:2, h in 1:8
        @test sum(f.decomp[i, h, :]) ≈ 1.0 atol=1e-14
        @test all(>=(0.0), f.decomp[i, h, :])
        @test all(<=(1.0), f.decomp[i, h, :])
    end
    # h=1 for the FIRST variable is entirely its own shock (P lower triangular)
    @test f.decomp[1, 1, 1] == 1.0
    @test f.decomp[1, 1, 2] == 0.0
end

@testset "fevd: all fixtures, rows sum to one" begin
    for (fx, p, h, tol) in [("var2_p1.csv",1,6,1e-11), ("var3_p2.csv",2,6,1e-11),
                            ("uschange.csv",2,6,1e-10), ("econ5.csv",2,4,1e-10),
                            ("var2_short.csv",2,4,1e-11)]
        m = fit_var(read_fixture(fx), p; deterministic=:const)
        f = fevd(m, h); K = size(f.decomp, 1)
        ref = load_ref_fevd(fx, p, h)
        @test f.decomp ≈ ref atol=tol
        for i in 1:K, hh in 1:h
            @test sum(f.decomp[i, hh, :]) ≈ 1.0 atol=1e-13
        end
    end
end

@testset "fevd: built from the orthogonalised IRF, consistently" begin
    m = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    h = 6; f = fevd(m, h); Psi = orth_ma_rep(m, h)
    # MSE contribution of shock j to variable i at horizon H is
    # sum_{m=0}^{H-1} Psi_m[i,j]^2, normalised by the row total
    for i in 1:3, H in 1:h
        num = [sum(Psi[mm+1][i,j]^2 for mm in 0:(H-1)) for j in 1:3]
        @test f.decomp[i, H, :] ≈ num ./ sum(num) atol=1e-12
    end
end
```

### 7.8 Bootstrap bands — reproducibility and coverage, not point values

```julia
@testset "bootstrap bands: seeded reproducibility" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    a = irf(m, 4; boot=true, runs=200, level=95.0, rng=MersenneTwister(4321))
    b = irf(m, 4; boot=true, runs=200, level=95.0, rng=MersenneTwister(4321))
    c = irf(m, 4; boot=true, runs=200, level=95.0, rng=MersenneTwister(9999))
    @test a.lower == b.lower && a.upper == b.upper      # bit-identical
    @test a.lower != c.lower
    # the band must bracket the point estimate
    for h in 1:5
        @test all(a.lower[h] .<= a.orth[h] .<= a.upper[h])
    end
    # a wider level gives a wider band
    w = irf(m, 4; boot=true, runs=200, level=99.0, rng=MersenneTwister(4321))
    @test all(w.upper[3] .>= a.upper[3])
    @test all(w.lower[3] .<= a.lower[3])
end

@testset "bootstrap bands: threaded == serial" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    p = irf(m, 4; boot=true, runs=200, rng=MersenneTwister(7), parallel=true)
    s = irf(m, 4; boot=true, runs=200, rng=MersenneTwister(7), parallel=false)
    @test p.lower == s.lower && p.upper == s.upper
end

@testset "bootstrap bands: coverage of a known IRF (slow; tagged)" begin
    # the band claims 95% coverage; check that, not a point value.
    # vars' bands cannot be matched numerically -- different scheme, different RNG.
    cov = irf_band_coverage(; reps=1000, n=200, level=95.0,
                            rng=MersenneTwister(20261013))   # test helper
    @test 0.92 <= cov <= 0.98
end
```

### 7.9 Refusals — both references mishandle the degenerate horizons

`[V]` Verbatim reference behaviour:

| Case | `vars` | statsmodels |
|---|---|---|
| `n.ahead = 0` | `subscript out of bounds` | **no error** — returns shape `(1,2,2)` |
| `n.ahead = -1` | `data is too long` | `IndexError: index 0 is out of bounds for axis 0 with size 0` |
| `fevd(n.ahead = 0)` | `subscript out of bounds` | **no error** — returns an empty `(2,0,2)` decomposition |
| unknown impulse name | `Please provide variables names in impulse that are in the set of endogenous variables.` | `IndexError` |

```julia
@testset "refusals" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)

    # h must be >= 1. statsmodels returns an EMPTY fevd for h=0; vars errors
    # with "subscript out of bounds". Neither is acceptable.
    @test_throws ArgumentError irf(m, 0)
    @test_throws ArgumentError irf(m, -1)
    @test_throws ArgumentError fevd(m, 0)
    @test_throws ArgumentError fevd(m, -3)
    e = try fevd(m, 0) catch err; err end
    @test occursin("horizon", lowercase(sprint(showerror, e)))

    # impulse / response selection by name
    @test_throws ArgumentError irf(m, 4; impulse=:nope)
    @test_throws ArgumentError irf(m, 4; response=:nope)
    @test_throws ArgumentError irf(m, 4; impulse=0)
    @test_throws ArgumentError irf(m, 4; impulse=3)

    # bootstrap arguments
    @test_throws ArgumentError irf(m, 4; boot=true, runs=0)
    @test_throws ArgumentError irf(m, 4; boot=true, level=0.0)
    @test_throws ArgumentError irf(m, 4; boot=true, level=100.0)
    @test_throws ArgumentError irf(m, 4; boot=true, level=95)     # Int, not Float64? decide and pin

    # an unstable VAR has no convergent long-run effect
    ex = fit_var(read_fixture("var2_explosive.csv"), 1; deterministic=:const)
    @test !is_stable(ex)
    @test_throws ArgumentError lr_effects(ex)
    e = try lr_effects(ex) catch err; err end
    @test occursin("stable", lowercase(sprint(showerror, e)))
    # but the finite-horizon IRF is still well defined, and must not refuse
    @test_nowarn irf(ex, 4)
end
```

**Note the last pair.** `[V]` Neither reference refuses a long-run
effect on an explosive VAR — statsmodels computes
`inv(I - sum A_j)` regardless, producing a number with no meaning. With
`var2_explosive`'s max root of 1.040418520983, the sum does not
converge. This is a new honest-refusal case: refuse `lr_effects` when
`!is_stable`, keep the finite-horizon IRF working.

### 7.10 Type stability, allocations, plot recipes

```julia
@testset "type stability and allocations" begin
    m = fit_var(read_fixture("uschange.csv"), 2; deterministic=:const)
    @test eltype(ma_rep(m, 6)) === Matrix{Float64}
    @test (@inferred ma_rep(m, 6)) isa Vector{Matrix{Float64}}
    @test (@inferred fevd(m, 6)) isa FEVDResult

    # the recursion allocates one K x K matrix per horizon and nothing more
    ma_rep(m, 20)                                  # warm up
    a = @allocated ma_rep(m, 20)
    @test a < 21 * 5 * 5 * 8 * 3                   # 3x the matrices themselves

    # irf cost must be linear in h, not quadratic
    t10 = @elapsed for _ in 1:50; ma_rep(m, 10); end
    t40 = @elapsed for _ in 1:50; ma_rep(m, 40); end
    @test t40 < 6 * t10
end

@testset "plot recipes" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    @test_nowarn RecipesBase.apply_recipe(Dict{Symbol,Any}(), irf(m, 6))
    @test_nowarn RecipesBase.apply_recipe(Dict{Symbol,Any}(), fevd(m, 6))
    @test_nowarn RecipesBase.apply_recipe(Dict{Symbol,Any}(),
                     irf(m, 6; boot=true, runs=50, rng=MersenneTwister(1)))
end
```

`[S]` Follow `test/test_recipes.jl`'s existing pattern for recipe tests —
the package already has recipes for `ACFResult`, `PeriodogramResult`, the
three decompositions and the two diagnostic-plot types, and `[S]`
**none for `Forecast`**. IRF and FEVD recipes are new work either way;
match the house style in `src/recipes.jl`.

---

## 8. Proposed Julia API

```julia
ma_rep(m::VARModel, h::Integer)       -> Vector{Matrix{Float64}}   # length h+1, Phi_i
orth_ma_rep(m::VARModel, h::Integer)  -> Vector{Matrix{Float64}}   # Psi_i = Phi_i P

struct IRFResult
    orth::Vector{Matrix{Float64}}          # length h+1, [response, impulse]
    nonorth::Vector{Matrix{Float64}}
    lower::Union{Nothing,Vector{Matrix{Float64}}}    # bootstrap only
    upper::Union{Nothing,Vector{Matrix{Float64}}}
    stderr_orth::Union{Nothing,Vector{Matrix{Float64}}}
    stderr_nonorth::Union{Nothing,Vector{Matrix{Float64}}}
    horizons::UnitRange{Int}               # 0:h
    names::Vector{Symbol}
    cumulative::Bool
    level::Union{Nothing,Float64}
    method::Symbol                         # :none | :analytic | :bootstrap
    runs::Union{Nothing,Int}
end

irf(m::VARModel, h::Integer;
    cumulative::Bool = false,
    impulse = nothing, response = nothing,      # Symbol / Int / vectors, or nothing for all
    stderr::Bool = false,                        # attach analytic SEs
    boot::Bool = false, runs::Integer = 100, level::Real = 95.0,
    rng::AbstractRNG = Random.default_rng(), parallel::Bool = true) -> IRFResult

irf_stderr(m::VARModel, h::Integer; orth::Bool=false, cumulative::Bool=false)
    -> Vector{Matrix{Float64}}
cum_effect_stderr(m::VARModel, h::Integer; orth::Bool=false)
lr_effects(m::VARModel; orth::Bool=false)   -> Matrix{Float64}
lr_effect_stderr(m::VARModel; orth::Bool=false)

struct FEVDResult
    decomp::Array{Float64,3}               # [variable, horizon, shock]
    horizons::UnitRange{Int}               # 1:h  -- NOT 0:h, see section 4(a)
    names::Vector{Symbol}
end

fevd(m::VARModel, h::Integer) -> FEVDResult

coef_alpha(m::VARModel) -> Vector{Float64}   # vec([A_1 ... A_p]); see section 5.1
```

`irf` with neither `stderr` nor `boot` returns point estimates only —
that is the common case and it should not pay for a covariance it was
not asked for. `method` records which uncertainty path, if any, produced
the bands, so a plot can label them correctly.

---

## 9. Implementation and performance

Per charter §7. The specifics for this item:

### 9.1 The recursion

```julia
function ma_rep(m::VARModel, h::Integer)
    K, p = size(m.A[1], 1), m.p
    Phi = [Matrix{Float64}(undef, K, K) for _ in 0:h]
    Phi[1] .= 0; @inbounds for i in 1:K; Phi[1][i,i] = 1.0; end
    @inbounds for i in 1:h
        acc = Phi[i+1]; fill!(acc, 0.0)
        for j in 1:min(i, p)
            mul!(acc, Phi[i-j+1], m.A[j], 1.0, 1.0)   # acc += Phi_{i-j} A_j
        end
    end
    return Phi
end
```

`mul!(C, A, B, alpha, beta)` accumulates without a temporary — that is
the whole trick, and it is why this allocates exactly `h+1` matrices and
nothing else. `[V]` For contrast, statsmodels hand-rolls a memo cache
(`self._g_memo`, with the comment *"be a bit cute to go faster"*) for the
matrix powers in its `G` construction; accumulating in the loop needs no
cache.

### 9.2 Analytic standard errors

`G_i` is built from the same recursion:

```
G_i = sum_{m=0}^{i-1} kron( ((A*)')^{i-1-m}[1:K, :] , Phi_m )
```

where `A*` is the `Kp × Kp` companion matrix (10.1's `companion`), and
`[1:K, :]` takes its first `K` rows. Then
`cov(vec(Phi_i)) = G_i Sigma_a G_i'`.

Three performance points:

1. **Accumulate the companion power in the loop.** `(A*')^k` for
   `k = 0 … i-1` is one `mul!` per step. Do not call `^` or recompute
   from scratch per horizon, and do not build a cache.
2. **Do not materialize the Kronecker product.** `kron(X, Phi_m)` is
   `K²p × K²` dense; the contraction `G Sigma_a G'` never needs it
   whole. Either accumulate `G` once per horizon (it is only
   `K² × K²p`, which is small) or, if `K` is large enough to matter,
   use `Kronecker.jl`'s lazy product — but **do not add the dependency
   for this item**; the dense `G` is `K² × K²p`, which for `K = 10, p = 4`
   is 100 × 400, trivially small.
3. **For the orthogonalized version** the second term needs `D⁺`, the
   duplication matrix pseudo-inverse. `[V]` It has a closed form:
   `D'D` is exactly diagonal with entries 1 and 2 (off-diagonal mass
   `0.0` for k = 2, 3, 5, 8), so `D⁺ = (D'D)⁻¹D'` and
   `max|pinv(D) − closed form| = 2.22e-16`. Both references call
   `pinv`. Do not.

The orthogonalized covariance has two terms — one through `A`, one
through `Sigma` — and `[S]` statsmodels' own source marks the textbook
statement as wrong at that equation (`# Lutkepohl typo, cov_sig
correct`). §5.2's bootstrap validates statsmodels' version, so follow
it, and **cite the bootstrap in the docstring** rather than the book.

### 9.3 Bootstrap

Residual resampling by row, `runs` independent replications,
`Threads.@threads` over them with **one RNG per replication** seeded
deterministically from the base seed — otherwise the threaded and serial
paths cannot agree bit-for-bit, and §7.8 asserts that they do. Each
replication needs its own `Z` buffer; do not share one.

---

## 10. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. Implement `ma_rep` / `orth_ma_rep` and `fevd` first — point
   estimates, both references agreeing, nothing uncertain. Get §7.1,
   §7.2, §7.3 and §7.7 green before touching standard errors.
3. Add `coef_alpha` and assert the flattening (§7.5) **before** writing
   `G`. A wrong flattening is plausible-looking and 40% off.
4. Implement the analytic standard errors; verify with the numerical
   Jacobian test (§7.6) before comparing against statsmodels, so the
   check is independent.
5. Implement the bootstrap. Verify by reproducibility and coverage, not
   against `vars`' numbers (§6).
6. IRF and FEVD plot recipes in `src/recipes.jl`, house style.
7. Run the battery. Target ≥ 250 assertions in `test/test_irf.jl`.
8. Copy `scripts/ref_R_irf.R`, `scripts/ref_py_irf.py` and
   `scripts/verify_irf_se.py` to `test/verification/irf/`.
9. `@code_warntype` on `ma_rep`, `irf`, `fevd`, `irf_stderr`; record the
   benchmark table for `k ∈ {2, 5, 10}`, `h ∈ {8, 24}`.
10. Manual section. Lead with the ordering warning from §4.1 — it is the
    thing users get wrong, and both references leave them to it.
11. Mark 10.3 `✅ built` in `development-sequence.md` with Stage-9-level
    detail, and add the new honest-refusal case (`lr_effects` on an
    unstable VAR) to the project's running list.
12. **Add to the charter §5.10** the finding that neither reference
    refuses a long-run effect on an explosive VAR.

---

## 11. Bundle additions for this item

```
fixtures/
  ref_R_irf.json        # 6 fixtures x {ortho, non-ortho, cumulative x2, Psi, Phi, FEVD},
                        #   plus a permuted-ordering case and a seeded vars bootstrap run
  ref_py_irf.json       # statsmodels counterparts, PLUS the analytic standard errors
                        #   (stderr, stderr_orth, cum_effect_stderr, lr_effect_stderr)
                        #   which vars does not provide at all
  verify_irf_se.json    # the independent verification output: numerical-Jacobian SEs
                        #   and 4000-replication parametric-bootstrap SDs
scripts/
  ref_R_irf.R           # regenerates ref_R_irf.json
  ref_py_irf.py         # regenerates ref_py_irf.json
  cmp_irf.py            # R-vs-Python comparison; reproduces every table in section 3
  verify_irf_se.py      # the two independent checks in section 5; ~1 minute
```
