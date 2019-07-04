module Impute

using IterTools
using Statistics
using StatsBase
using Tables: Tables, materializer, istable

import Base.Iterators

export impute, impute!, chain, chain!, drop, drop!, interp, interp!, ImputeError

function __init__()
    sym = join(["chain", "chain!", "drop", "drop!", "interp", "interp!"], ", ", " and ")

    @warn(
        """
        The following symbols will not be exported in future releases: $sym.
        Please qualify your calls with `Impute.<method>(...)` or explicitly import the symbol.
        """
    )
end

"""
    ImputeError{T} <: Exception

Is thrown by `impute` methods when the limit of imputable values has been exceeded.

# Fields
* msg::T - the message to print.
"""
struct ImputeError{T} <: Exception
    msg::T
end

Base.showerror(io::IO, err::ImputeError) = println(io, "ImputeError: $(err.msg)")

include("context.jl")
include("imputors.jl")

const global imputation_methods = Dict{Symbol, Type}(
    :drop => Drop,
    :interp => Interpolate,
    :fill => Fill,
    :locf => LOCF,
    :nocb => NOCB,
)

"""
    impute!(data, method::Symbol=:interp, args...; limit::Float64=0.1)

Looks up the `Imputor` type for the `method`, creates it and calls
`impute!(imputor::Imputor, data, limit::Float64)` with it.

# Arguments
* `data`: the datset containing missing elements we should impute.
* `method::Symbol`: the imputation method to use
    (options: [`:drop`, `:fill`, `:interp`, `:locf`, `:nocb`])
* `args::Any...`: any arguments you should pass to the `Imputor` constructor.
* `limit::Float64`: missing data ratio limit/threshold (default: 0.1)
"""
function impute!(data, method::Symbol, args...; limit::Float64=0.1)
    imputor_type = imputation_methods[method]
    imputor = length(args) > 0 ? imputor_type(args...) : imputor_type()
    return impute!(imputor, data, limit)
end

"""
    impute!(data, missing::Function, method::Symbol=:interp, args...; limit::Float64=0.1)

Creates the appropriate `Imputor` type and `Context` (using `missing` function) in order to call
`impute!(imputor::Imputor, ctx::Context, data)` with them.

# Arguments
* `data`: the datset containing missing elements we should impute.
* `missing::Function`: the missing data function to use
* `method::Symbol`: the imputation method to use
    (options: [`:drop`, `:fill`, `:interp`, `:locf`, `:nocb`])
* `args::Any...`: any arguments you should pass to the `Imputor` constructor.
* `limit::Float64`: missing data ratio limit/threshold (default: 0.1)
"""
function impute!(data, missing::Function, method::Symbol, args...; limit::Float64=0.1)
    imputor_type = imputation_methods[method]
    imputor = length(args) > 0 ? imputor_type(args...) : imputor_type()
    return Context(; limit=limit, is_missing=missing)() do ctx
        impute!(imputor, ctx, data)
    end
end

"""
    impute(data, args...; kwargs...)

Copies the `data` before calling `impute!(new_data, args...; kwargs...)`
"""
function impute(data, args...; kwargs...)
    return impute!(deepcopy(data), args...; kwargs...)
end

"""
    chain!(data, missing::Function, imputors::Imputor...; kwargs...)

Creates a `Chain` with `imputors` and calls `impute!(imputor, missing, data; kwargs...)`
"""
function chain!(data, missing::Function, imputors::Imputor...; kwargs...)
    imputor = Chain(imputors...)
    return impute!(imputor, missing, data; kwargs...)
end

"""
    chain!(data, imputors::Imputor...; kwargs...)

Creates a `Chain` with `imputors` and calls `impute!(imputor, data; kwargs...)`
"""
function chain!(data, imputors::Imputor...; kwargs...)
    imputor = Chain(imputors...)
    return impute!(imputor, data; kwargs...)
end

"""
    chain(data, args...; kwargs...)

Copies the `data` before calling `chain!(data, args...; kwargs...)`
"""
function chain(data, args...; kwargs...)
    result = deepcopy(data)
    return chain!(data, args...; kwargs...)
end

"""
    drop!(data; limit=1.0)

Utility method for `impute!(data, :drop; limit=limit)`
"""
drop!(data; limit=1.0) = impute!(data, :drop; limit=limit)

"""
    drop(data; limit=1.0)

Utility method for `impute(data, :drop; limit=limit)`
"""
Iterators.drop(data; limit=1.0) = impute(data, :drop; limit=limit)

"""
    interp!(data; limit=1.0)

Utility method for `impute!(data, :interp; limit=limit)`
"""
interp!(data; limit=1.0) = impute!(data, :interp; limit=limit)

"""
    interp(data; limit=1.0)

Utility method for `impute(data, :interp; limit=limit)`
"""
interp(data; limit=1.0) = impute(data, :interp; limit=limit)

end  # module
