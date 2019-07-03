"""
    Chain <: Imputor

Runs multiple `Imputor`s on the same data in the order they're provided.

# Fields
* `imputors::Array{Imputor}`
"""
struct Chain <: Imputor
    imputors::Vector{Imputor}
end

"""
    Chain(imputors::Imputor...) -> Chain

Creates a Chain using the `Imputor`s provided (ordering matters).
"""
Chain(imputors::Imputor...) = Chain(collect(imputors))

"""
    impute!(imp::Chain, missing::Function, data; limit::Float64=0.1)

Creates a `Context` and runs the `Imputor`s on the supplied data.

# Arguments
* `imp::Chain`: the chain to run
* `missing::Function`: the missing function to use in the `Context` to pass to the `Imputor`s
* `data`: our data to impute
* `limit::Float64`: the missing data ration limit/threshold

# Returns
* our imputed data
"""
function impute!(imp::Chain, missing::Function, data; limit::Float64=0.1)
    context = Context(; limit=limit, is_missing=missing)

    for imputor in imp.imputors
        data = impute!(imputor, context, data)
    end

    return data
end

"""
    impute!(imp::Chain, data; limit::Float64=0.1)


Infers the missing data function from the `data` and passes that to
`impute!(imp::Chain, missing::Function, data; limit::Float64=0.1)`.

# Arguments
* `imp::Chain`: the chain to run
* `data`: our data to impute
* `limit::Float64`: the missing data ration limit/threshold

# Returns
* our imputed data
"""
function impute!(imp::Chain, data; limit::Float64=0.1)
    impute!(imp, ismissing, data; limit=limit)
end
