# The `@example` trap, as a build failure rather than a reviewer's memory.
#
# Documenter shows a block's printed output OR the display of its last
# expression, not both: a block that prints and then ends in a plot (or any
# other displayed value) renders the plot and silently drops what it printed,
# so the prose quotes numbers the reader never sees. The fix is to split the
# block -- printing in one, plotting in the next, same named session -- or to
# end it with `nothing # hide`. (Stage 9/10 documentation: 41 blocks had to
# be split after the fact; handoff/stage 11/stage-11-documentation-handoff.md
# §8 asks for this check.)
#
# Each block is parsed with Julia's own parser, so a `println` inside a
# string or a comment is not mistaken for a print.

const _PRINTS = (:println, :print, :show, :display, :printstyled)

_hasprint(x) = x isa Expr && ((x.head === :call && x.args[1] in _PRINTS) ||
                              (x.head === :macrocall && x.args[1] === Symbol("@show")) ||
                              any(_hasprint, x.args))

# is the block's last expression displayed? (`nothing`, a print, a loop or an
# import displays nothing; anything else -- a plot, a value, an assignment --
# does)
function _displays(e)
    e === nothing && return false
    e isa Symbol && return e !== :nothing
    e isa Expr || return true
    h = e.head
    # `a; b` on one line is a block: what shows is its last expression
    if h in (:block, :toplevel)
        rest = filter(a -> !(a isa LineNumberNode), e.args)
        return isempty(rest) ? false : _displays(rest[end])
    end
    h === :call && (e.args[1] in _PRINTS || e.args[1] === :foreach) && return false
    h === :macrocall && e.args[1] === Symbol("@show") && return false
    h in (:for, :while, :using, :import, :const) && return false
    return true
end

"""
    check_example_blocks(dir) -> Vector{String}

Every `@example` block under `dir` that prints and whose last expression is
displayed, as `"file:line"` of its opening fence.
"""
function check_example_blocks(dir::AbstractString)
    bad = String[]
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".md") || continue
        path = joinpath(root, f)
        L = readlines(path; keep=false)
        i = 1
        while i <= length(L)
            if startswith(L[i], "```@example")
                j = i + 1
                while j <= length(L) && !startswith(L[j], "```")
                    j += 1
                end
                code = join(L[(i + 1):(j - 1)], "\n")
                ex = try
                    Meta.parseall(code)
                catch
                    nothing
                end
                if ex !== nothing && _hasprint(ex)
                    last = nothing
                    for a in ex.args
                        a isa LineNumberNode || (last = a)
                    end
                    lastline = isempty(strip(code)) ? "" : strip(split(rstrip(code), "\n")[end])
                    suppressed = endswith(lastline, ";") || occursin("# hide", lastline) && last === :nothing
                    (!suppressed && _displays(last)) && push!(bad, "$(relpath(path, dir)):$i")
                end
                i = j + 1
            else
                i += 1
            end
        end
    end
    return bad
end
