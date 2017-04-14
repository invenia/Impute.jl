"""
    Chain <: Imputor

Runs multiple `Imputor`s on the same data in the order they're provided.

# Fields
* `imputors::Array{Imputor}`
"""
type Chain <: Imputor
    imputors::Array{Imputor}
end

"""
    Chain(imputors::Imputor...) -> Chain

Creates a Chain using the `Imputor`s provided (ordering matters).
"""
Chain(imputors::Imputor...) = Chain(collect(imputors))

"""
    impute!(imp::Chain, missing::Function, data::Dataset; limit::Float64=0.1)

Creates a `Context` and runs the `Imputor`s on the supplied data.

# Arguments
* `imp::Chain`: the chain to run
* `missing::Function`: the missing function to use in the `Context` to pass to the `Imputor`s
* `data::Dataset`: our data to impute
* `limit::Float64`: the missing data ration limit/threshold

# Returns
* `Dataset`: our imputed data
"""
function impute!(imp::Chain, missing::Function, data::Dataset; limit::Float64=0.1)
    ctx = Context(*(size(data)...), 0, limit, missing)

    for imputor in imp.imputors
        impute!(imputor, copy(ctx), data)
    end

    return data
end

"""
    impute!(imp::Chain, data::Dataset; limit::Float64=0.1)


Infers the missing data function from the `data` and passes that to
`impute!(imp::Chain, missing::Function, data::Dataset; limit::Float64=0.1)`.

# Arguments
* `imp::Chain`: the chain to run
* `data::Dataset`: our data to impute
* `limit::Float64`: the missing data ration limit/threshold

# Returns
* `Dataset`: our imputed data
"""
function impute!(imp::Chain, data::Dataset; limit::Float64=0.1)
    f = if isa(data, Union{NullableArray, DataTable})
        isnull
    elseif isa(data, Union{DataArray, DataFrame})
        isna
    else
        isnan
    end

    impute!(imp, f, data; limit=limit)
end
