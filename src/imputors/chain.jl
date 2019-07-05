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
    impute!(imp::Chain, data)

Runs the `Imputor`s on the supplied data.

# Arguments
* `imp::Chain`: the chain to run
* `data`: our data to impute

# Returns
* our imputed data
"""
function impute!(imp::Chain, data)
    for imputor in imp.imputors
        data = impute!(imputor, data)
    end

    return data
end
