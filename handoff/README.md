# handoff/

Per-task detail backing specific rows in `development-sequence.md`. Where
`CLAUDE.md` (repo root) holds evergreen project-wide policy, files here
hold the verified reference signatures, design rationale, implementation,
and hand-checked test values for one specific piece of functionality.

## Naming

`stage-<N>-<short-name>.md`, where `<N>` is the exact stage number from
`development-sequence.md` (e.g. `stage-1.2-filters.md`,
`stage-6.5-arma-mle.md`). Keeps a 1:1 mapping to the roadmap and sorts
naturally in roadmap order.

## Template for a new handoff doc

```markdown
# Handoff: Stage <N> — <short description>

Status: pending | implemented ✅

## Where this fits
- Depends on: ...
- Feeds into: ...

## Verified reference: R
<exact signature, verified via web search or Rscript, not from memory>

## Verified reference: Python
<same>

## Proposed Julia API
<function name, signature, design rationale for any naming choices>

## Implementation
<code, once written>

## Hand-verified test values
<expected outputs computed independently -- Python/R script, or by-hand
arithmetic -- before being written as Julia test assertions, especially
important when no Julia runtime is available to check against directly>
```

## Why keep these after implementation

Docstrings capture *what* a function does. These capture *why* an
argument is named or ordered the way it is, what was checked against
what, and what was explicitly rejected as a reference (e.g. the
`pandas.rolling` correction in `stage-1.2-filters.md`) — worth keeping
even once the code and tests exist, since that reasoning doesn't live
anywhere else.

Mark `Status: implemented ✅` at the top once done rather than deleting.
