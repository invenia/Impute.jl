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
            "Declare Missings" => "api/declaremissings.md",
            "Validation" => "api/validation.md",
            "Filtering" => "api/filtering.md",
            "Imputation" => "api/imputation.md",
            "Chaining" => "api/chaining.md",
        ],
    ],
    repo="https://github.com/invenia/Impute.jl/blob/{commit}{path}#L{line}",
    sitename="Impute.jl",
    authors="Invenia Technical Computing Corporation",
    strict=true,
    checkdocs=:exports,
)

deploydocs(
    repo = "github.com/invenia/Impute.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
