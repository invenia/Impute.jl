"""
    AbstractContext

An imputation context records summary information about missing data for an imputation algorithm.
"""
abstract type AbstractContext end

"""
    ismissing(ctx::AbstractContext, x) -> Bool

Uses `ctx.is_missing` to determine if x is missing. If x is a named tuple or an abstract array
then `ismissing` will return true if `ctx.is_missing` returns true for any element.
The ctx.count is increased whenever whenever we return true and if `ctx.count / ctx.num`
exceeds our `ctx.limit` we throw an `ImputeError`

# Arguments
* `ctx::Context`: the contextual information about missing information.
* `x`: the value to check (may be an single values, abstract array or row)
"""
function Base.ismissing(ctx::AbstractContext, x)
    missing = if isa(x, NamedTuple)
        any(entry -> ctx.is_missing(entry[2]), pairs(x))
    elseif isa(x, AbstractArray)
        any(ctx.is_missing, x)
    else
        ctx.is_missing(x)
    end

    missing_update!(ctx, missing)

    return missing
end

"""
    findfirst(ctx::AbstractContext, data::AbstractVector) -> Int

Returns the first not missing index in `data`.

# Arguments
* `ctx::AbstractContext`: the context to pass into `ismissing`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the first index in `data` that isn't missing
"""
function Base.findfirst(ctx::AbstractContext, data::AbstractVector)
    return findfirst(x -> !ismissing(ctx, x), data)
end

"""
    findlast(ctx::AbstractContext, data::AbstractVector) -> Int

Returns the last not missing index in `data`.

# Arguments
* `ctx::AbstractContext`: the context to pass into `ismissing`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the last index in `data` that isn't missing
"""
function Base.findlast(ctx::AbstractContext, data::AbstractVector)
    return findlast(x -> !ismissing(ctx, x), data)
end

"""
    findnext(ctx::AbstractContext, data::AbstractVector) -> Int

Returns the next not missing index in `data`.

# Arguments
* `ctx::AbstractContext`: the context to pass into `ismissing`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the next index in `data` that isn't missing
"""
function Base.findnext(ctx::AbstractContext, data::AbstractVector, idx::Int)
    return findnext(x -> !ismissing(ctx, x), data, idx)
end

"""
    Context

Records base information about the missing data and assume all observations are equally
weighted.

# Fields
* `n::Int`: number of observations
* `count::Int`: number of missing values found
* `limit::Float64`: allowable limit for missing values to impute
* `is_missing::Function`: returns a Bool if the value counts as missing
* `on_complete::Function`: a function to run when imputation is complete
"""
mutable struct Context <: AbstractContext
    num::Int
    count::Int
    limit::Float64
    is_missing::Function
    on_complete::Function
end

function Context(;
    limit::Float64=1.0,
    is_missing::Function=ismissing,
    on_complete::Function=complete
)
    Context(0, 0, limit, is_missing, on_complete)
end

function (ctx::Context)(f::Function)
    _ctx = copy(ctx)
    _ctx.num = 0
    _ctx.count = 0

    result = f(_ctx)
    ctx.on_complete(_ctx)
    return result
end

Base.copy(x::Context) = Context(x.num, x.count, x.limit, x.is_missing, x.on_complete)

function missing_update!(ctx::Context, miss)
    ctx.num += 1

    if miss
        ctx.count += 1
    end
end

function complete(ctx::Context)
    if (ctx.count / ctx.num) > ctx.limit
        throw(ImputeError(
            "More than $(ctx.limit * 100)% of values were missing ()."
        ))
    end
end


"""
    WeightedContext

Records information about the missing data relative to a set of weights.

# Fields
* `num::Int`: number of observations
* `s::Float64`: sum of missing values weights
* `limit::Float64`: allowable limit for missing values to impute
* `is_missing::Function`: returns a Bool if the value counts as missing
* `on_complete::Function`: a function to run when imputation is complete
* `wv::AbstractWeights`: a set of statistical weights to use when evaluating the importance
  of each observation
"""
mutable struct WeightedContext <: AbstractContext
    num::Int
    s::Float64
    limit::Float64
    is_missing::Function
    on_complete::Function
    wv::AbstractWeights
end

function WeightedContext(
    wv::AbstractWeights;
    limit::Float64=1.0,
    is_missing::Function=ismissing,
    on_complete::Function=complete
)
    WeightedContext(0, 0.0, limit, is_missing, on_complete, wv)
end

function (ctx::WeightedContext)(f::Function)
    _ctx = copy(ctx)
    _ctx.num = 0
    _ctx.s = 0.0

    result = f(_ctx)
    ctx.on_complete(_ctx)
    return result
end

function Base.copy(x::WeightedContext)
    WeightedContext(x.num, x.s, x.limit, x.is_missing, x.on_complete, wv)
end

function missing_update!(ctx::WeightedContext, miss)
    ctx.num += 1

    if miss
        ctx.s += ctx.wv[ctx.num]
    end
end

function complete(ctx::WeightedContext)
    if (ctx.s / sum(ctx.wv)) > ctx.limit
        throw(ImputeError(
            "More than $(ctx.limit * 100)% of weighted values were missing ()."
        ))
    end
end
