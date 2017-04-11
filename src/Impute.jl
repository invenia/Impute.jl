module Impute

import NullableArrays: NullableArray
import DataTables: DataTable

import DataArrays: DataArray, isna
import DataFrames: DataFrame

import RDatasets: dataset

export impute, chain, ImputeError, interp, interp!

immutable ImputeError{T} <: Exception
    msg::T
end

Base.showerror(io::IO, err::ImputeError) = println(io, "ImputeError: $(err.msg)")

"""
    Context

Stores common summary information for all Imputor types.

# Fields
* `num::Int`: number of observations
* `count::Int`: number of missing values found
* `limit::Float64`: allowable limit for missing values to impute
* `missing::Function`: returns a Bool if the value counts as missing.
"""
type Context
    num::Int
    count::Int
    limit::Float64
    missing::Function
end

Context(limit::Float64, missing::Function) = Context(0, 0, limit, missing)

Base.copy(x::Context) = Context(x.num, x.count, x.limit, x.missing)

function is_missing(ctx::Context, x)
    if ctx.missing(x)
        ctx.count += 1

        if (ctx.count / ctx.num) > ctx.limit
            throw(ImputeError(
                "More than $(ctx.limit * 100)% of values were missing."
            ))
        end

        return true
    else
        return false
    end
end

abstract Imputor

function Base.findfirst{T<:Any}(imp::Imputor, data::AbstractArray{T, 1})
    return findfirst(x -> !is_missing(imp.ctx, x), data)
end

function Base.findlast{T<:Any}(imp::Imputor, data::AbstractArray{T, 1})
    return findlast(x -> !is_missing(imp.ctx, x), data)
end

function Base.findnext{T<:Any}(imp::Imputor, data::AbstractArray{T, 1}, idx::Int)
    return findnext(x -> !is_missing(imp.ctx, x), data, idx)
end

type Drop <: Imputor
    ctx::Context
end

function (imp::Drop){T<:Any}(data::AbstractArray{T, 1})
    imp.ctx.num = length(data)
    imp.ctx.count = 0
    return filter!(x -> !is_missing(imp.ctx, x), data)
end

type LOCF <: Imputor
    ctx::Context
end

function (imp::LOCF){T<:Any}(data::AbstractArray{T, 1})
    imp.ctx.num = length(data)
    imp.ctx.count = 0

    start_idx = findfirst(imp, data) + 1
    for i in start_idx:length(data)
        if is_missing(imp.ctx, data[i])
            data[i] = data[i-1]
        end
    end

    return data
end

type NOCB <: Imputor
    ctx::Context
end

function (imp::NOCB){T<:Any}(data::AbstractArray{T, 1})
    imp.ctx.num = length(data)
    imp.ctx.count = 0

    end_idx = findlast(imp, data) - 1
    for i in end_idx:-1:1
        if is_missing(imp.ctx, data[i])
            data[i] = data[i+1]
        end
    end

    return data
end

type Interpolate <: Imputor
    ctx::Context
end

function (imp::Interpolate){T<:Any}(data::AbstractArray{T, 1})
    imp.ctx.num = length(data)
    imp.ctx.count = 0

    i = findfirst(imp, data) + 1
    while i < length(data)
        if is_missing(imp.ctx, data[i])
            prev_idx = i - 1
            next_idx = findnext(imp, data, i + 1)

            if next_idx > 0
                gap_sz = (next_idx - prev_idx) - 1

                diff = data[next_idx] - data[prev_idx]
                incr = diff / T(gap_sz + 1)
                start_val = data[prev_idx]
                stop_val = data[next_idx]

                values = Real(start_val):Real(incr):Real(stop_val)

                idx_range = prev_idx:(prev_idx + length(values) - 1)
                # println(collect(idx_range))
                # println(values)

                data[idx_range] = values
                i = next_idx
            else
                break
            end
        end
        i += 1
    end

    return data
end

type Fill <: Imputor
    ctx::Context
    value::Any
end

Fill(ctx::Context; value=mean) = Fill(ctx, value)

function (imp::Fill){T<:Any}(data::AbstractArray{T, 1})
    imp.ctx.num = length(data)
    imp.ctx.count = 0

    fill_val = if isa(imp.value, Function)
        imp.value(drop(copy(data)))
    else
        imp.value
    end

    imp.ctx.count = 0
    for i in 1:length(data)
        if is_missing(imp.ctx, data[i])
            data[i] = fill_val
        end
    end

    return data
end


# TODO: support multi-dimensional data

const global imputation_methods = Dict{Symbol, Type}(
    :drop => Drop,
    :interp => Interpolate,
    :fill => Fill,
    :locf => LOCF,
    :nocb => NOCB,
)


"""
    impute(data::AbstractArray, missing::Function, method::Symbol, limit=0.05)

Scans the `AbstractArray` and replaces all missing entries using the specified method.
If too much data is missing `impute` with throw and error.

# Arguments
* `data::AbstractArray`: the array that may contain missing elements, which should be imputed.
* `missing::Function`:
"""
function impute!(data::AbstractArray, missing::Function; method::Symbol=:interp, limit::Float64=0.1, kwargs...)
    imputor = imputation_methods[method](Context(limit, missing); kwargs...)
    return imputor(data)
end

impute(data::AbstractArray, missing::Function; kwargs...) = impute!(copy(data), missing; kwargs...)

impute!(data::NullableArray; kwargs...) = impute!(data, isnull; kwargs...)

impute!(data::DataArray; kwargs...) = impute!(data, isna; kwargs...)

impute!{T<:Real}(data::AbstractArray{T}; kwargs...) = impute!(data, isnan; kwargs...)


function impute!(df::Union{DataFrame, DataTable}; kwargs...)
    for col in names(df)
        impute!(df[col]; kwargs...)
    end
    return df
end

impute(data::Union{AbstractArray, DataFrame, DataTable}; kwargs...) = impute!(copy(data); kwargs...)

function chain!(data::Union{AbstractArray, DataFrame, DataTable}, imputors::Imputor...)
    for imp in imputors
        impute!(result, imp)
    end
    return result
end

function chain(data::Union{AbstractArray, DataFrame, DataTable}, imputors::Imputor...)
    result = deepcopy(data)
    return chain!(data, imputors...)
end

Base.convert{T<:Any}(::Type{T}, x::Nullable) = convert(T, x.value)
drop!(data::AbstractArray; limit=1.0) = impute!(data; method=:drop, limit=limit)
Base.drop(data::AbstractArray; limit=1.0) = drop!(copy(data); limit=limit)
interp!(data::AbstractArray; limit=1.0) = impute!(data; method=:interp, limit=limit)
interp(data::AbstractArray; limit=1.0) = interp!(copy(data); limit=limit)

include("utils.jl")

end  # module
