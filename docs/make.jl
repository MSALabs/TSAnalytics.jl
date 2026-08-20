using Documenter
using TSAnalytics

DocMeta.setdocmeta!(TSAnalytics, :DocTestSetup, :(using TSAnalytics); recursive=true)

makedocs(;
    modules=[TSAnalytics],
    authors="Mousum Dutta",
    sitename="TSAnalytics.jl",
    format=Documenter.HTML(;
        canonical="https://MSALabs.github.io/TSAnalytics.jl",
        edit_link="main",
        assets=String[],
        # api.md is one page for every exported symbol across the whole
        # package and keeps growing stage by stage (186 KiB at Stage 8.2,
        # 199 KiB at 8.3, 205 KiB at 8.4) -- raised well past the default
        # 200 KiB hard limit so the build doesn't start failing purely on
        # page size as more stages land; size_threshold_warn left at the
        # default so a genuinely oversized page still gets flagged.
        size_threshold=1_000_000,
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md",
    ],
    # doctest=:fix locally regenerates expected doctest output when you
    # deliberately change behaviour; leave as default (true) in CI so a
    # docstring example silently drifting out of date fails the build.
    doctest=true,
    checkdocs=:exports,  # build fails if an exported name has no docstring
)

deploydocs(;
    repo="github.com/MSALabs/TSAnalytics.jl",
    devbranch="main",
)
