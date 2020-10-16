using Documenter, Impute

makedocs(
    modules=[Impute],
    format=Documenter.HTML(assets=["assets/invenia.css"]),
    pages=[
        "Home" => "index.md",
        "Walkthroughs" => [
            "Spatiotemporal" => "walkthroughs/spatiotemporal.md",
            "SVD" => "walkthroughs/svd.md",
        ],
        "API" => [
            "Impute" => "api/impute.md",
            "Validators" => "api/validators.md",
            "Filter" => "api/filter.md",
            "Imputors" => "api/imputors.md",
            "Chain" => "api/chain.md",
            "Functional" => "api/functional.md",
            "Utilities" => "api/utils.md",
        ],
    ],
    repo="https://github.com/invenia/Impute.jl/blob/{commit}{path}#L{line}",
    sitename="Impute.jl",
    authors="Invenia Technical Computing Corporation",
)

deploydocs(
    repo = "github.com/invenia/Impute.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
