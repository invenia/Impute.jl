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
    impute!(data, imp::Chain)

Runs the `Imputor`s on the supplied data.

# Arguments
* `imp::Chain`: the chain to run
* `data`: our data to impute

# Returns
* our imputed data
"""
function impute!(data, imp::Chain)
    for imputor in imp.imputors
        data = impute!(data, imputor)
    end

    return data
end
