# Handoff: Stage 10.4b — Cointegration: the Phillips-Ouliaris test

**Status: implemented ✅ (2026-10-13).**

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first**, then
`stage-10.4a-engle-granger-handoff.md` §1, which explains why row 10.4
splits in two and should not be re-litigated here.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`.

---

## 1. Where this fits

- **Depends on:** `_ols` (`src/unitroot.jl:51`), `_bartlett_lrv`
  (`:595` — the Newey-West long-run variance, already built and
  verified for KPSS), `_interp_pvalue_right` (`:559`), 10.0's
  `tsmatrix`, and **new Phillips-Ouliaris critical-value tables** (§4).
- **Does not depend on** 10.1, 10.3 or 10.4a. Fully parallel.
- **This is the only item in Stage 10 with a single software reference
  for everything.** statsmodels has no Phillips-Ouliaris test at all, so
  `tseries` and `urca` are both R, and they implement *different
  statistics* (§3). Plan the verification accordingly.

---

## 2. Scope

### In

- The **Z-type** residual-based cointegration tests: `Pu` (the variance
  ratio) and `Pz` (the trace statistic), per Phillips & Ouliaris (1990).
- `tseries::po.test`'s own statistic, which is a third thing again (§3).
- Critical values for `m = 2 … 6` and three deterministic cases.
- Honest refusal outside that coverage, and for the capped p-values.

### Out

- The tau-type Engle-Granger test — 10.4a.
- Johansen (10.5), VECM (10.6).
- Hansen's parameter-instability tests, which `urca` does not ship.

---

## 3. Three statistics, not one — establish which before coding

`[V]` On `coint2_yes.csv` (250 × 2, cointegrated):

| Reference / option | statistic | p-value or critical values |
|---|---|---|
| `tseries::po.test(demean=FALSE, lshort=TRUE)` | −38.461465058 | p = 0.0100 (**capped**), lag 2 |
| `tseries::po.test(demean=FALSE, lshort=FALSE)` | −42.497104431 | p = 0.0100, lag 8 |
| `tseries::po.test(demean=TRUE, lshort=TRUE)` | **−128.400260380** | p = 0.0100, lag 2 |
| `tseries::po.test(demean=TRUE, lshort=FALSE)` | −133.570179045 | p = 0.0100, lag 8 |
| `urca::ca.po(type="Pu", demean="none")` | 64.711116825 | cval 20.3933, 25.9711, 38.3413 |
| `urca::ca.po(type="Pu", demean="constant")` | 143.609807496 | 27.8536, 33.7130, 48.0021 |
| `urca::ca.po(type="Pu", demean="trend")` | 146.377767632 | 41.2488, 48.8439, 65.1714 |
| `urca::ca.po(type="Pz", demean="none")` | 69.952592382 | 33.9267, 40.8217, 55.1911 |
| `urca::ca.po(type="Pz", demean="constant")` | 71.864578112 | 47.5877, 55.2202, 71.9273 |
| `urca::ca.po(type="Pz", demean="trend")` | 72.048173790 | 71.9586, 81.3812, 102.0167 |

Three separate families: `po.test`'s signed Z statistic, `ca.po`'s `Pu`
and `ca.po`'s `Pz`. **Decide what TSAnalytics ships before writing
anything**, and name it unambiguously in the API.

**Recommendation: implement `Pu` and `Pz` from `urca::ca.po`.** They are
the statistics in Phillips & Ouliaris (1990), they come with proper
critical-value tables rather than a capped p-value, and they generalize
to `m > 2`. Expose `po.test`'s variant only if a user asks for it, and
if so say in the docstring that it is `tseries`' own normalization.

### 3.1 The `demean` choice moves the answer enormously

`[V]` On `coint2_trend.csv` (a trend inside the cointegrating relation)
the conclusion flips:

```
po.test(demean=FALSE, lshort=TRUE)   stat =   -1.841599638   p = 0.1500  -> no rejection
po.test(demean=TRUE,  lshort=TRUE)   stat =  -28.403486608   p = 0.0100  -> rejection
ca.po(Pu, demean="none")             stat =    3.186717998   cval 20.39/25.97/38.34  -> no
ca.po(Pu, demean="constant")         stat =   50.239046798   cval 27.85/33.71/48.00  -> yes at 5%
ca.po(Pu, demean="trend")            stat =  170.020050757   cval 41.25/48.84/65.17  -> yes at 1%
```

`[V]` And on `coint2_yes.csv`, `po.test` moves from −38.46 to −128.40
between `demean=FALSE` and `TRUE` — a factor of 3.3.

**Decision:** `deterministic` is a required keyword with **no default**,
or a default of `:const` that the docstring flags loudly. A silent
default here is a correctness trap, not a convenience. Also note
`[V]` `ca.po`'s `demean` values are `"none" | "constant" | "trend"` while
`po.test`'s is a `Bool` — map both into the charter §5.2 vocabulary
(`:none`, `:const`, `:trend`) and document the mapping.

### 3.2 `lshort` is a bandwidth choice, and it matters

`[V]` `po.test`'s `lshort=TRUE` uses lag 2 on 250 observations;
`lshort=FALSE` uses lag 8. The statistic moves from −38.461465058 to
−42.497104431 (`demean=FALSE`) and −128.400260380 to −133.570179045
(`demean=TRUE`).

`[S]` `_bartlett_lrv(u, l)` already exists in `src/unitroot.jl:595`,
built and verified for KPSS. **Reuse it** and expose `bandwidth` as an
integer keyword with the two named shortcuts, rather than a `Bool`
whose meaning is invisible at the call site.

---

## 4. The tables: `m = 2 … 6` only, and the source is a paper

`[S]` `urca/R/ca-po.R:66` and `:76` hardcode the critical values as two
`c(5, 3, 3)` arrays — `m − 1` indexing over five rows (so **m = 2 … 6**),
three significance levels, three deterministic cases. `[V]` And `ca.po`
refuses beyond that, cleanly:

```
ca.po(matrix with 7 columns)  ->  "Please provide a matrix with at least
                                   two and maximal six columns"
ca.po(matrix with 6 columns)  ->  works (teststat 113.329421)
```

**Licensing (charter §4): statsmodels has no Phillips-Ouliaris test, so
there is no BSD source to transcribe from.** `urca` is GPL and must not
be copied. The numbers must come from **Phillips & Ouliaris (1990),
Econometrica 58(1), Tables Ia–IIc**, cited in the source comment, with
`urca`'s values used only as the check.

`[V]` The values to check against, read off `ca.po` output (10%, 5%, 1%):

| type | deterministic | m=2 | m=3 |
|---|---|---|---|
| `Pu` | none | 20.3933, 25.9711, 38.3413 | 26.7022, 32.9392, 46.4097 |
| `Pu` | constant | 27.8536, 33.7130, 48.0021 | 33.6955, 40.5252, 53.8731 |
| `Pu` | trend | 41.2488, 48.8439, 65.1714 | 46.1061, 53.8300, 69.2629 |
| `Pz` | none | 33.9267, 40.8217, 55.1911 | 62.1436, 71.2751, 89.6679 |
| `Pz` | constant | 47.5877, 55.2202, 71.9273 | 80.2034, 89.7619, 109.4525 |
| `Pz` | trend | 71.9586, 81.3812, 102.0167 | 113.4929, 124.3933, 145.8644 |

`[S]` The full `c(5,3,3)` arrays are visible in `urca/R/ca-po.R:66,76`
for cross-checking all five `m` values; `ref_R_coint.json` records the
subsets actually exercised. **Transcribe from the paper, verify against
these.**

### 4.1 No p-values — and `po.test`'s are censored

`[V]` `ca.po` reports critical values only, no p-value. `[V]`
`tseries::po.test` reports a p-value but it is **interval-censored at
the table bounds**: every rejecting case above returns exactly `0.0100`
and every non-rejecting one exactly `0.1500`, with R emitting
*"p-value smaller than printed p-value"* as a warning.

**Decision:** report **critical values only**, and have `pvalue` refuse
via the charter §6.4 contract extension — the same treatment 10.5 needs
for Johansen and 10.4a needs beyond its table coverage. Do **not**
return a clipped p-value. `[S]` This is the established pattern: KPSS's
clipped p-values are already one of the project's six documented
honest-refusal cases.

---

## 5. Reference values to assert

`[V]` All from `urca::ca.po`, 250 observations.

**`coint2_yes.csv`** (m=2, cointegrated) — §3's table.

**`coint2_no.csv`** (m=2, two independent random walks, null true):

```
Pu none 1.580876338   constant 4.713093767    trend 24.250664499
Pz none 13.201022728  constant 13.169762299   trend 12.583691175
po.test demean=FALSE lshort  -0.299428899  p 0.1500
po.test demean=TRUE  lshort  -3.255932123  p 0.1500
```

Every one well below its 10% critical value — the test has correct size
here, and that is worth an assertion.

**`coint3_r1.csv`** (m=3, one cointegrating relation):

```
Pu none 157.207598308  constant 157.187097065  trend 163.586553113
Pz none 237.231461141  constant 236.596735708   trend 234.324548712
po.test demean=FALSE lshort -188.195634733, llong -187.772204127
po.test demean=TRUE  lshort -188.179937898, llong -188.045770693
```

**`coint2_trend.csv`** (m=2, trend in the relation) — §3.1's table, plus:

```
Pz none 15.742507795  constant 21.157473443  trend 19.081914891
po.test demean=FALSE llong  -1.157139320  p 0.1500
po.test demean=TRUE  llong -39.621572369  p 0.0100
```

---

## 6. Proposed Julia API

```julia
struct PhillipsOuliarisTest <: HypothesisTest
    statistic::Float64
    critical_values::NamedTuple       # (; levels = (10.0, 5.0, 1.0), values)
    kind::Symbol                      # :Pu | :Pz
    deterministic::Symbol             # :none | :const | :trend
    m::Int                            # number of series
    bandwidth::Int
    nobs::Int
    beta::Vector{Float64}
    resid::Vector{Float64}
    pvalue_available::Bool            # always false -- see section 4.1
end

phillips_ouliaris_test(Y;
    kind::Symbol = :Pz,
    deterministic::Symbol = :const,          # NO silent default behaviour: see 3.1
    bandwidth::Union{Nothing,Integer,Symbol} = :short)  # :short | :long | an Int
        -> PhillipsOuliarisTest
```

`pvalue(t)` throws an `ArgumentError` naming `critical_values`. A
convenience `reject(t; level=5.0) -> Bool` is worth adding here, because
without a p-value the comparison is the whole interface, and leaving
every caller to index `critical_values` invites the off-by-one that
`[V]` the 10%/5%/1% ordering (ascending, unlike the ADF tables'
1%/5%/10%) makes easy.

**Note the ordering difference.** `[V]` `ca.po`'s `cval` is
`(10%, 5%, 1%)` — ascending critical values, right-tail test — while
`[V]` `mackinnoncrit` in 10.4a is `(1%, 5%, 10%)`. Store `levels`
explicitly in the NamedTuple rather than relying on position, and assert
the ordering in a test.

---

## 7. Test battery

Suggested file: `test/test_phillips_ouliaris.jl`. Target **≥ 140
assertions**. References in `fixtures/ref_R_coint.json`.

### 7.1 `Pu` and `Pz` against `urca::ca.po`

```julia
@testset "phillips_ouliaris_test: Pu and Pz vs urca::ca.po" begin
    cases = [
     ("coint2_yes.csv",   :Pu, :none,   64.711116825,  (20.3933, 25.9711, 38.3413)),
     ("coint2_yes.csv",   :Pu, :const, 143.609807496,  (27.8536, 33.7130, 48.0021)),
     ("coint2_yes.csv",   :Pu, :trend, 146.377767632,  (41.2488, 48.8439, 65.1714)),
     ("coint2_yes.csv",   :Pz, :none,   69.952592382,  (33.9267, 40.8217, 55.1911)),
     ("coint2_yes.csv",   :Pz, :const,  71.864578112,  (47.5877, 55.2202, 71.9273)),
     ("coint2_yes.csv",   :Pz, :trend,  72.048173790,  (71.9586, 81.3812, 102.0167)),
     ("coint2_no.csv",    :Pu, :none,    1.580876338,  (20.3933, 25.9711, 38.3413)),
     ("coint2_no.csv",    :Pu, :const,   4.713093767,  (27.8536, 33.7130, 48.0021)),
     ("coint2_no.csv",    :Pu, :trend,  24.250664499,  (41.2488, 48.8439, 65.1714)),
     ("coint2_no.csv",    :Pz, :none,   13.201022728,  (33.9267, 40.8217, 55.1911)),
     ("coint2_no.csv",    :Pz, :const,  13.169762299,  (47.5877, 55.2202, 71.9273)),
     ("coint2_no.csv",    :Pz, :trend,  12.583691175,  (71.9586, 81.3812, 102.0167)),
     ("coint3_r1.csv",    :Pu, :none,  157.207598308,  (26.7022, 32.9392, 46.4097)),
     ("coint3_r1.csv",    :Pu, :const, 157.187097065,  (33.6955, 40.5252, 53.8731)),
     ("coint3_r1.csv",    :Pu, :trend, 163.586553113,  (46.1061, 53.8300, 69.2629)),
     ("coint3_r1.csv",    :Pz, :none,  237.231461141,  (62.1436, 71.2751, 89.6679)),
     ("coint3_r1.csv",    :Pz, :const, 236.596735708,  (80.2034, 89.7619, 109.4525)),
     ("coint3_r1.csv",    :Pz, :trend, 234.324548712,  (113.4929, 124.3933, 145.8644)),
     ("coint2_trend.csv", :Pu, :none,    3.186717998,  (20.3933, 25.9711, 38.3413)),
     ("coint2_trend.csv", :Pu, :const,  50.239046798,  (27.8536, 33.7130, 48.0021)),
     ("coint2_trend.csv", :Pu, :trend, 170.020050757,  (41.2488, 48.8439, 65.1714)),
     ("coint2_trend.csv", :Pz, :none,   15.742507795,  (33.9267, 40.8217, 55.1911)),
     ("coint2_trend.csv", :Pz, :const,  21.157473443,  (47.5877, 55.2202, 71.9273)),
     ("coint2_trend.csv", :Pz, :trend,  19.081914891,  (71.9586, 81.3812, 102.0167))]

    for (fx, kind, det, stat, cv) in cases
        t = phillips_ouliaris_test(read_fixture(fx); kind=kind, deterministic=det)
        @test t.statistic ≈ stat rtol=1e-8
        @test t.critical_values.levels == (10.0, 5.0, 1.0)     # ASCENDING, unlike ADF
        @test all(isapprox.(t.critical_values.values, cv; atol=1e-4))
        @test t.m == size(read_fixture(fx), 2)
        @test t.kind == kind && t.deterministic == det
    end
end
```

### 7.2 Size and power

```julia
@testset "size and power" begin
    for kind in (:Pu, :Pz), det in (:none, :const, :trend)
        # cointegrated -> reject at 1%
        t = phillips_ouliaris_test(read_fixture("coint2_yes.csv"); kind=kind, deterministic=det)
        @test reject(t; level=1.0)
        t3 = phillips_ouliaris_test(read_fixture("coint3_r1.csv"); kind=kind, deterministic=det)
        @test reject(t3; level=1.0)
        # independent random walks -> do not reject at 10%
        n = phillips_ouliaris_test(read_fixture("coint2_no.csv"); kind=kind, deterministic=det)
        @test !reject(n; level=10.0)
    end
end
```

### 7.3 The deterministic choice changes the conclusion

```julia
@testset "the deterministic choice is not cosmetic" begin
    Y = read_fixture("coint2_trend.csv")      # a trend sits INSIDE the relation
    none  = phillips_ouliaris_test(Y; kind=:Pu, deterministic=:none)
    const_= phillips_ouliaris_test(Y; kind=:Pu, deterministic=:const)
    trend = phillips_ouliaris_test(Y; kind=:Pu, deterministic=:trend)
    @test !reject(none;   level=10.0)         #   3.19 vs 20.39
    @test  reject(const_; level=5.0)          #  50.24 vs 33.71
    @test  reject(trend;  level=1.0)          # 170.02 vs 65.17
    # the statistics must be far apart, not nearly equal
    @test trend.statistic > 50 * none.statistic
end
```

### 7.4 Bandwidth

```julia
@testset "bandwidth: :short, :long and explicit" begin
    Y = read_fixture("coint2_yes.csv")
    s = phillips_ouliaris_test(Y; kind=:Pz, deterministic=:const, bandwidth=:short)
    l = phillips_ouliaris_test(Y; kind=:Pz, deterministic=:const, bandwidth=:long)
    @test s.bandwidth == 2          # tseries' lshort rule at n = 250
    @test l.bandwidth == 8          # its llong rule
    @test s.statistic != l.statistic
    @test phillips_ouliaris_test(Y; kind=:Pz, deterministic=:const,
                                 bandwidth=2).statistic ≈ s.statistic atol=1e-12
    # bandwidth 0 is the un-corrected statistic, and must be allowed
    @test_nowarn phillips_ouliaris_test(Y; kind=:Pz, deterministic=:const, bandwidth=0)
    @test_throws ArgumentError phillips_ouliaris_test(Y; bandwidth=-1)
    @test_throws ArgumentError phillips_ouliaris_test(Y; bandwidth=250)
    @test_throws ArgumentError phillips_ouliaris_test(Y; bandwidth=:bogus)
end

@testset "reuses _bartlett_lrv rather than a second implementation" begin
    # the long-run variance must be the already-verified KPSS helper
    u = randn(MersenneTwister(3), 200)
    @test TSAnalytics._bartlett_lrv(u, 0) ≈ sum(abs2, u)/200 atol=1e-12
    @test TSAnalytics._bartlett_lrv(u, 4) != TSAnalytics._bartlett_lrv(u, 0)
end
```

### 7.5 Coverage refusals and no p-value

```julia
@testset "coverage: m = 2..6 only" begin
    # urca refuses m > 6 with a clear message; the table genuinely stops there
    Y7 = randn(MersenneTwister(7), 250, 7)
    for j in 2:7; Y7[:, j] .= cumsum(Y7[:, j]); end
    @test_throws ArgumentError phillips_ouliaris_test(Y7; kind=:Pz, deterministic=:const)
    e = try phillips_ouliaris_test(Y7) catch err; err end
    @test occursin("six", lowercase(sprint(showerror, e))) ||
          occursin("6", sprint(showerror, e))
    # m = 6 must work
    @test_nowarn phillips_ouliaris_test(Y7[:, 1:6]; kind=:Pz, deterministic=:const)
    # m = 1 is not a cointegration test
    @test_throws ArgumentError phillips_ouliaris_test(Y7[:, 1:1])
end

@testset "no p-value is reported, and none is faked" begin
    t = phillips_ouliaris_test(read_fixture("coint2_yes.csv"); kind=:Pz, deterministic=:const)
    @test !t.pvalue_available
    @test_throws ArgumentError pvalue(t)
    e = try pvalue(t) catch err; err end
    @test occursin("critical_values", sprint(showerror, e))
    # tseries::po.test reports 0.0100 and 0.1500 -- the table bounds, censored.
    # We must not return either as if it were a p-value.
    @test !hasproperty(t, :pvalue) || isnan(getfield(t, :pvalue)) == false
end
```

### 7.6 Contract, printing, performance

```julia
@testset "contract, show, type stability" begin
    t = phillips_ouliaris_test(read_fixture("coint3_r1.csv"); kind=:Pu, deterministic=:const)
    @test t isa HypothesisTest
    @test statistic(t) === t.statistic
    @test isconcretetype(PhillipsOuliarisTest)
    @test (@inferred phillips_ouliaris_test(read_fixture("coint2_yes.csv"))) isa PhillipsOuliarisTest
    s = sprint(show, t)
    @test occursin("Phillips", s)
    @test occursin("Pu", s)                       # which statistic must be visible
    @test occursin("const", lowercase(s))          # and which deterministic case
    # reject() must read levels by name, not by position
    @test reject(t; level=5.0) isa Bool
    @test_throws ArgumentError reject(t; level=2.5)
end
```

---

## 8. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. **Decide and record** which statistic ships (§3 recommends `Pu` and
   `Pz`), before writing code.
3. Transcribe the critical values **from Phillips & Ouliaris (1990),
   Tables Ia–IIc**, citing the paper in the source comment. Verify
   against the `[V]` values in §4 — do not copy from `urca` (GPL).
4. Implement in `src/cointegration.jl` alongside 10.4a, reusing
   `_bartlett_lrv` and `_ols`.
5. Extend `HypothesisTest` per charter §6.4 so `pvalue` can refuse, and
   add `reject(t; level)`.
6. Run the battery. Target ≥ 140 assertions.
7. Copy the verification scripts to `test/verification/coint/` (shared
   with 10.4a).
8. **Add to the charter §5.10**: `tseries::po.test`'s p-values are
   interval-censored at 0.01 / 0.15 with a warning, and its `demean`
   flag changes the statistic by a factor of 3.3 on a cointegrated
   fixture.
9. Manual section: the Z-versus-tau distinction, the `demean`
   sensitivity with the `coint2_trend` example from §3.1, and the
   bandwidth choice.

---

## 9. Bundle additions

Shares 10.4a's fixtures and scripts; no new files. The
Phillips-Ouliaris values live in `fixtures/ref_R_coint.json` under
`po_yes`, `po_no`, `po_r1` and `po_trend`, each holding four
`po.test` variants and six `ca.po` variants.
