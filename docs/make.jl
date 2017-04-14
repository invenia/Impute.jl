using Documenter, Impute, RDatasets

makedocs(
    modules=[Impute],
    format=:html,
    pages=[
        "Home" => "index.md",
        "Impute" => "api/impute.md",
        "Context" => "api/context.md",
        "Imputors" => "api/imputors.md",
        "Utilities" => "api/utils.md",
    ],
    repo="https://github.com/invenia/Impute.jl/blob/{commit}{path}#L{line}",
    sitename="Impute.jl",
    authors="Invenia Technical Computing Corporation",
    assets=["assets/invenia.css"],
)

deploydocs(
    repo = "github.com/invenia/Impute.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
