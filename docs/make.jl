using Documenter, Impute

makedocs(
    modules=[Impute],
    format=:html,
    pages=[
        "Home" => "index.md",
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
