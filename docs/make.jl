using Documenter
using TSAnalytics

DocMeta.setdocmeta!(TSAnalytics, :DocTestSetup, :(using TSAnalytics); recursive=true)

makedocs(;
    modules=[TSAnalytics],
    authors="Mousum Dutta",
    sitename="TSAnalytics.jl",
    format=Documenter.HTML(;
        canonical="https://<your-org>.github.io/TSAnalytics.jl",
        edit_link="main",
        assets=String[],
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
    repo="github.com/<your-org>/TSAnalytics.jl",
    devbranch="main",
)
