export partrans, invpartrans

"""
    partrans(raw::AbstractVector{<:Real}) -> Vector{Float64}

Map unconstrained real parameters (as used by an optimizer) to AR/MA
coefficients guaranteed to lie in the stationary/invertible region, via
the Monahan (1984) transform: `tanh` to squash each raw parameter into a
partial autocorrelation in `(-1,1)`, then the Durbin-Levinson recursion
to convert those PACF values into AR (or MA) coefficients.

Matches R's `stats::arima` C implementation (`arima.c`'s `partrans`)
exactly -- verified against R's actual source and independently
re-derived numerically (not just transcribed), see
`handoff/stage-4.2-monahan-handoff.md`. **Deliberately does NOT match**
Python's `statsmodels.tsa.statespace.tools.constrain_stationary_univariate`,
which implements the same Monahan (1984) result via a genuinely different
squashing function (`u/sqrt(1+u^2)`, plus a sign negation) -- both are
individually valid parameterizations of the same constrained space, not a
bug in either. Concretely: raw (unconstrained, optimizer-space) parameter
values are **not** expected to match between an R-based and a
Python-based implementation, even fitting the identical model to
identical data -- only the final *constrained* AR/MA coefficients (where
the likelihood is actually maximized) are comparable across conventions.

Element-type-generic (via `eltype(raw)`, not hardcoded `Float64`) so a
`ForwardDiff.Dual`-typed `raw` -- as `_optimize` (Stage 4.1) passes when
`fit_arma`'s objective calls this from inside the optimizer -- flows
through without a `MethodError`; confirmed by direct testing (Stage 6.5),
since this had no caller inside an AD-differentiated closure before then.

# Examples
```jldoctest
julia> using TSAnalytics

julia> partrans([0.5, -0.3, 0.2])
3-element Vector{Float64}:
  0.654235633768312
 -0.4090939097636928
  0.197375320224904
```
"""
function partrans(raw::AbstractVector{<:Real})
    p = length(raw)
    p == 0 && return Float64[]
    new = tanh.(collect(eltype(raw), raw))
    work = copy(new)
    for j in 2:p
        a = new[j]
        for k in 1:(j-1)
            work[k] -= a * new[j-k]
        end
        new[1:j-1] .= work[1:j-1]
    end
    return new
end

"""
    invpartrans(phi::AbstractVector{<:Real}) -> Vector{Float64}

Inverse of [`partrans`](@ref): map AR/MA coefficients back to the
unconstrained space, for initializing an optimizer from a closed-form
starting estimate (e.g. OLS or Yule-Walker) rather than an arbitrary
point. Matches R's `invpartrans` (`arima.c`) exactly, verified against
source and by round-tripping through [`partrans`](@ref).

# Examples
```jldoctest
julia> using TSAnalytics

julia> invpartrans(partrans([0.5, -0.3, 0.2]))
3-element Vector{Float64}:
  0.5
 -0.3
  0.2
```
"""
function invpartrans(phi::AbstractVector{<:Real})
    p = length(phi)
    p == 0 && return Float64[]
    new = collect(eltype(phi), phi)
    work = copy(new)
    for j in p:-1:2
        a = new[j]
        for k in 1:(j-1)
            work[k] = (new[k] + a*new[j-k]) / (1 - a^2)
        end
        new[1:j-1] .= work[1:j-1]
    end
    return atanh.(new)
end
