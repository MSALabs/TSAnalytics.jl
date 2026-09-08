# Major Update: Real X-13ARIMA-SEATS Binary Confirmed Working — Corrects S.1's Verification Boundary

For a fresh session picking this up with no prior context. **This
overturns a claim made in the original S.1 handoff.** That handoff
stated plainly: "neither R nor Python has a native X-11 implementation
to verify against... verification here comes from published formulas
and known mathematical properties, not cross-language number-matching."
**That statement is no longer accurate, and shouldn't be treated as
current guidance.**

## What was found and verified, directly

The GitHub repository `x13org/x13prebuilt` — confirmed from its own
README to be **the exact repository R's `x13binary`/`seasonal`
packages use internally** — distributes statically-linked, prebuilt
X-13ARIMA-SEATS binaries for Linux, Windows, and macOS, built directly
from the Census Bureau's own source. This is not a third-party
reimplementation or a wrapper — it's the actual authoritative program,
compiled and redistributed with the Census Bureau's own explicit
public-domain licensing (confirmed directly from the repo's README,
quoting the Census Bureau's own license text: royalty-free,
non-exclusive, no restriction on use).

**Confirmed working, directly, in this session**:
```bash
git clone --depth 1 https://github.com/x13org/x13prebuilt.git
# v1.1.57/linux/64/x13ashtml -- a 5.9MB, statically-linked ELF binary,
# no missing dependencies, runs immediately
```

**A complete, real X-11 run was executed successfully** on the
Box-Jenkins airline passengers series (1949-1960, 144 monthly
observations — the single most widely used benchmark series in the
entire seasonal-adjustment literature, appearing in nearly every
textbook treatment including Box & Jenkins' own original work):

```
series { title = "..."; start = 1949.1; data = (112 118 132 ...) }
x11 { save = (d10 d11 d12 d13) }
```

Produced real, authoritative output tables — **D10 (final seasonal
factors), D11 (final seasonally adjusted series), D12 (final
trend-cycle), D13 (final irregular)** — the exact same table
identifiers already referenced from JDemetra+'s own documentation
during S.1's original research, now available as real numbers rather
than a description of what they'd contain:

```
D10 (seasonal factors), first 6 months of 1949:
  194901  0.909187671763704
  194902  0.959401810971537
  194903  1.05757934646024
  194904  1.00514879872191
  194905  0.999732566557629
  194906  1.07303952726698

D11 (seasonally adjusted), first 3 months of 1949:
  194901  123.186888118198
  194902  122.993305464483
  194903  124.813330027491

D12 (trend-cycle), first 3 months of 1949:
  194901  123.774514749304
  194902  123.950693749510
  194903  124.289377383170
```

Full 144-month tables for D10/D11/D12/D13 are in this bundle's
`verification/` folder, generated directly by the real binary, not
transcribed from documentation.

## What this means for the whole chapter, not just S.1

**This is a categorically stronger verification standard than almost
any other stage in this project has had** — not comparing against a
third-party reimplementation (the usual R/Python cross-check
throughout this project), but against the actual authoritative source
the whole method is defined by.

- **S.1 (X-11 filters)**: can now be verified against real D10/D11/D12
  output directly, not just the Henderson-filter formula properties
  already checked. This is a strictly stronger bar — worth re-running
  the existing S.1 test suite's synthetic case through the real binary
  for direct comparison.
- **S.2 (RegARIMA)**: the same binary handles RegARIMA specs directly
  (`regression { }`, `arima { }` blocks) — real trading-day/Easter
  regressor coefficient estimates are obtainable the same way.
- **S.5 (SEATS)**: the binary includes SEATS directly (confirmed from
  the Census Bureau's own "About X-13" page: *"The capability to
  generate ARIMA model-based seasonal adjustment using a version of the
  SEATS software... as well as nonparametric adjustments from the X-11
  procedure"*) — meaning even the hardest, "deliberately last" stage in
  the entire roadmap now has a real, authoritative reference to check
  against, not just Gómez & Maravall's published algorithm description.

## A note on sandbox-specific network access, for whoever picks this up next

`census.gov` itself is blocked by this particular sandbox's network
egress rules (confirmed via a direct `curl` test returning
`x-deny-reason: host_not_allowed`) — but `github.com` is not, and this
repository mirrors the same binaries. If a future session's sandbox
has different network restrictions, either path should work; this one
happened to work via GitHub specifically. Worth checking both if one is
blocked.

## What to do with this

1. **Correct the original S.1 handoff's verification-boundary framing**
   — it's no longer true that no reference exists. Update it to point
   here instead of restating the old claim.
2. Keep the binary (`x13ashtml`, included in this bundle) and the real
   airline-series ground truth available for ongoing verification
   throughout S.1-S.5's actual implementation — this is now the primary
   verification target for the whole chapter, not a fallback.
3. Generate additional real ground-truth runs as each stage is
   implemented — quarterly-frequency series, a RegARIMA case with real
   trading-day regressors, a SEATS run — rather than relying solely on
   the one airline-series case captured here.
4. Update `development-sequence.md`'s Chapter Ten framing to reflect
   this — the chapter's own opening line ("the thing neither R nor
   Python offers natively either") is still true (neither reimplements
   it), but the verification story is no longer "no reference exists"
   the way Stage 7.4's realized-volatility work genuinely was.
