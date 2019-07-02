"""
    Context

Stores common summary information for all Imputor types.

# Fields
* `num::Int`: number of observations
* `count::Int`: number of missing values found
* `limit::Float64`: allowable limit for missing values to impute
* `missing::Function`: returns a Bool if the value counts as missing.
"""
mutable struct Context
    num::Int
    count::Int
    limit::Float64
    missing::Function
end

Context(limit::Float64, missing::Function=ismissing) = Context(0, 0, limit, missing)

Base.copy(x::Context) = Context(x.num, x.count, x.limit, x.missing)

"""
    ismissing(ctx::Context, x) -> Bool

Uses `ctx.missing` to determine if x is missing. If x is a named tuple or an abstract array
then `ismissing` will return true if `ctx.missing` returns true for any element.
The ctx.count is increased whenever whenever we return true and if `ctx.count / ctx.num`
exceeds our `ctx.limit` we throw an `ImputeError`

# Arguments
* `ctx::Context`: the contextual information about missing information.
* `x`: the value to check (may be an single values, abstract array or row)
"""
function Base.ismissing(ctx::Context, x)
    missing = if isa(x, NamedTuple)
        any(entry -> ctx.missing(entry[2]), pairs(x))
    elseif isa(x, AbstractArray)
        any(ctx.missing, x)
    else
        ctx.missing(x)
    end

    if missing
        ctx.count += 1

        if (ctx.count / ctx.num) > ctx.limit
            throw(ImputeError(
                "More than $(ctx.limit * 100)% of values were missing ()."
            ))
        end

        return true
    else
        return false
    end
end

"""
    findfirst(ctx::Context, data::AbstractVector) -> Int

Returns the first not missing index in `data`.

# Arguments
* `ctx::Context`: the context to pass into `ismissing`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the first index in `data` that isn't missing
"""
function Base.findfirst(ctx::Context, data::AbstractVector)
    return findfirst(x -> !ismissing(ctx, x), data)
end

"""
    findlast(ctx::Context, data::AbstractVector) -> Int

Returns the last not missing index in `data`.

# Arguments
* `ctx::Context`: the context to pass into `ismissing`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the last index in `data` that isn't missing
"""
function Base.findlast(ctx::Context, data::AbstractVector)
    return findlast(x -> !ismissing(ctx, x), data)
end

"""
    findnext(ctx::Context, data::AbstractVector) -> Int

Returns the next not missing index in `data`.

# Arguments
* `ctx::Context`: the context to pass into `ismissing`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the next index in `data` that isn't missing
"""
function Base.findnext(ctx::Context, data::AbstractVector, idx::Int)
    return findnext(x -> !ismissing(ctx, x), data, idx)
end
