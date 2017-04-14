"""
    Imputor

An imputor stores information about imputing values in `AbstractArray`s, `DataFrame`s and
`DataTable`s. New imputation methods are expected to sutype `Imputor` and, at minimum,
implement the `impute!{T<:Any}(imp::<MyImputor>, ctx::Context, data::AbstractArray{T, 1})`
method.
"""

abstract Imputor

"""
    impute!(imp::Imputor, data::Dataset, limit::Float64=0.1)

Creates a `Context` using information about `data`. These include

1. missing data function:
* `isnull`: if `NullableArray` or `DataTable`
* `isna`: if `DataArray` or `DataFrame`
* `isnan`: for anything else.

2. number of elements: `*(size(data)...)`

# Arguments
* `imp::Imputor`: the Imputor method to use
* `data::Dataset`: the data to impute
* `limit::Float64: missing data ratio limit/threshold (default: 0.1)`

# Return
* `Dataset`: the input `data` with values imputed.
"""
function impute!(imp::Imputor, data::Dataset, limit::Float64=0.1)
    f = if isa(data, Union{NullableArray, DataTable})
        isnull
    elseif isa(data, Union{DataArray, DataFrame})
        isna
    else
        isnan
    end

    ctx = Context(*(size(data)...), 0, limit, f)
    return impute!(imp, ctx, data)
end

"""
    impute!{T<:Any}(imp::Imputor, ctx::Context, data::AbstractArray{T, 2})

Imputes the data in a matrix by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `ctx::Context`: the contextual information for missing data
* `data::AbstractArray{T, 2}`: the data to impute

# Returns
* `AbstractArray{T, 2}`: the input `data` with values imputed
"""
function impute!{T<:Any}(imp::Imputor, ctx::Context, data::AbstractArray{T, 2})
    for i in 1:size(data, 2)
        impute!(imp, ctx, view(data, :, i))
    end
    return data
end

"""
    impute!{T<:Any}(imp::Imputor, ctx::Context, data::Table)

Imputes the data in a DataFrame or DataTable by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `ctx::Context`: the contextual information for missing data
* `data::Table`: the data to impute

# Returns
* `Table`: the input `data` with values imputed
"""
function impute!(imp::Imputor, ctx::Context, data::Table)
    for col in names(data)
        impute!(imp, ctx, data[col])
    end
    return data
end

imputors_path = joinpath(dirname(@__FILE__), "imputors")

for file in ("drop.jl", "locf.jl", "nocb.jl", "interp.jl", "fill.jl", "chain.jl")
    include(joinpath(imputors_path, file))
end
