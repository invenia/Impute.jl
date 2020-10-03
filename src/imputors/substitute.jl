"""
    Substitute(; statistic=nothing)
    Substitute(; robust=true, weights=nothing)

Substitute missing values with a summary statistic over the non-missing values.

# Keyword Arguments
* `statistic`: A summary statistic function to be applied to the non-missing values.
  This function should return a value of the same type as the input data `eltype`.
  If this function isn't passed in then the `defaultstats` function is used to make a
  best guess.
* `robust`: Whether to use `median` or `mean` for continuous datasets in `defaultstats`
* `weights`: A set of statistical weights to apply to the `mean` or `median` in `defaultstats`.

# Default Rules
Our default substitution rules defined in `defaultstats` are as follows:

* `mode` applies to non-`Real`s, `Bool`s, and `Integers` with few unique values.
* `median` is used for all other `Real` values that aren't restricted by the above rules.
  Optionally, `mean` can be used if `robust=false`. If statistical `weights` are passed in
  then a weighted `mean`/`median` will be calculated.

# Example
```jldoctest
julia> using Impute: Substitute, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Substitute(; statistic=mean); dims=2)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.66667  2.66667  5.0
 1.1  2.2  3.3      3.025    5.5
```
"""
struct Substitute <: Imputor
    statistic::Function
end

function Substitute(;
    statistic::Union{Function, Nothing}=nothing,
    robust=true,
    weights=nothing
)
    statistic !== nothing && return Substitute(statistic)

    return Substitute() do data
        if weights === nothing
            items = collect(skipmissing(data))
            defaultstats(items, robust)
        else
            mask = .!ismissing.(data)
            items = disallowmissing(data[mask])
            wv = weights[mask]
            # @show items robust wv
            defaultstats(items, robust, wv)
        end
    end
end

function _impute!(data::AbstractArray{Union{T, Missing}}, imp::Substitute) where T
    x = imp.statistic(data)
    return Base.replace!(data, missing => x)
end

# Auxiliary functions defining our default substitution rules

# If we're operating over Bools then we're probably better off using mode
defaultstats(data::AbstractArray{<:Bool}, robust::Bool, args...) = mode(data)

# If we're operating over integers with relatively few unique values then we're
# likely dealing with either counts or a categorical coding, so mode is probably
# safer
function defaultstats(data::AbstractArray{T}, robust::Bool, args...) where T <: Integer
    threshold = 0.25 * length(data)
    nunique = length(unique(data))
    nunique < threshold && return mode(data)
    result = robust ? median(data, args...) : mean(data, args...)
    return round(T, result)
end

# For most real valued data we should use median
function defaultstats(data::AbstractArray{<:Real}, robust::Bool, args...)
    return robust ? median(data, args...) : mean(data, args...)
end

# Fallback to mode as many types won't support mean or median anyways
defaultstats(data::AbstractArray, args...) = mode(data)
