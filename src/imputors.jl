"""
    Imputor

An imputor stores information about imputing values in `AbstractArray`s and `Tables.table`s.
New imputation methods are expected to sutype `Imputor` and, at minimum,
implement the `impute!{T<:Any}(imp::<MyImputor>, ctx::Context, data::AbstractArray{T, 1})`
method.
"""

abstract type Imputor end

"""
    impute!(imp::Imputor, data, limit::Float64=0.1)

Creates a `Context` using information about `data`. These include

1. missing data function which defaults to `missing`

2. number of elements: `*(size(data)...)`

# Arguments
* `imp::Imputor`: the Imputor method to use
* `data`: the data to impute
* `limit::Float64: missing data ratio limit/threshold (default: 0.1)`

# Return
* the input `data` with values imputed.
"""
function impute!(imp::Imputor, data, limit::Float64=0.1)
    Context(; limit=limit)() do ctx
        return impute!(imp, ctx, data)
    end
end

"""
    impute!(imp::Imputor, ctx::Context, data::AbstractMatrix)

Imputes the data in a matrix by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `ctx::Context`: the contextual information for missing data
* `data::AbstractMatrix`: the data to impute

# Returns
* `AbstractMatrix`: the input `data` with values imputed
"""
function impute!(imp::Imputor, ctx::Context, data::AbstractMatrix)
    for i in 1:size(data, 2)
        impute!(imp, ctx, view(data, :, i))
    end
    return data
end

"""
    impute!(imp::Imputor, ctx::Context, table)

Imputes the data in a table by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `ctx::Context`: the contextual information for missing data
* `table`: the data to impute

# Returns
* the input `data` with values imputed
"""
function impute!(imp::Imputor, ctx::Context, table)
    @assert istable(table)
    # Extract a columns iterate that we should be able to use to mutate the data.
    # NOTE: Mutation is not guaranteed for all table types, but it avoid copying the data
    columntable = Tables.columns(table)

    for cname in propertynames(columntable)
        impute!(imp, ctx, getproperty(columntable, cname))
    end

    return table
end


for file in ("drop.jl", "locf.jl", "nocb.jl", "interp.jl", "fill.jl", "chain.jl")
    include(joinpath("imputors", file))
end
