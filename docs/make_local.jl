# Local-only docs preview. `make.jl` (the real build script, used by CI)
# relies on Documenter auto-detecting the git remote for source-code
# links -- that only works when this directory is an actual git checkout.
# This repo isn't (yet) a git repository locally, so `make.jl` fails here
# with "Unable to automatically determine remote for main repo" even
# though nothing is actually wrong. This script is identical except it
# passes `remotes=nothing` explicitly and skips `deploydocs` (which only
# makes sense against a real git remote anyway). Once this directory
# becomes a real git repo, `make.jl` itself will work locally too and
# this file is no longer needed.
using Documenter
using TSAnalytics

DocMeta.setdocmeta!(TSAnalytics, :DocTestSetup, :(using TSAnalytics); recursive=true)

makedocs(;
    modules=[TSAnalytics],
    authors="Mousum Dutta",
    sitename="TSAnalytics.jl",
    format=Documenter.HTML(; assets=String[]),
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md",
    ],
    doctest=true,
    checkdocs=:exports,
)

println("\nBuilt to docs/build/index.html -- open it in a browser.")
