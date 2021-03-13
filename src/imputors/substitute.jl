"""
    Substitute(; statistic=Impute.defaultstats)

Substitute missing values with a summary statistic over the non-missing values.

# Keyword Arguments
* `statistic`: A summary statistic function to be applied to the non-missing values.
  This function should return a value of the same type as the input data `eltype`.
  If this function isn't passed in then the [`Impute.defaultstats`](@ref) function is used to make
  a best guess.

# Example
```jldoctest
julia> using Statistics; using Impute: Substitute, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Substitute(); dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.0  2.0   5.0
 1.1  2.2  3.3  2.75  5.5

julia> impute(M, Substitute(; statistic=mean); dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.66667  2.66667  5.0
 1.1  2.2  3.3      3.025    5.5
```
"""
struct Substitute{F<:Function} <: Imputor
    statistic::F
end

Substitute(; statistic=defaultstats) = Substitute(statistic)

function _impute!(data::AbstractArray{Union{T, Missing}}, imp::Substitute) where T
    mask = .!ismissing.(data)
    x = imp.statistic(disallowmissing(data[mask]))
    return Base.replace!(data, missing => x)
end


"""
    WeightedSubstitute(; statistic=Impute.defaultstats, weights)

Substitute missing values with a weighted summary statistic over the non-missing values.

# Keyword Arguments
* `statistic`: A summary statistic function to be applied to the non-missing values.
  This function should return a value of the same type as the input data `eltype`.
  If this function isn't passed in then the [`Impute.defaultstats`](@ref) function is used to make
  a best guess.
* `weights`: A set of statistical weights to pass to the `statistic` function.

# Example
```jldoctest
julia> using Statistics, StatsBase; using Impute: WeightedSubstitute, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> wv = weights([0.5, 0.2, 0.3, 0.1, 0.4]);

julia> impute(M, WeightedSubstitute(; weights=wv); dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.75  2.75     5.0
 1.1  2.2  3.3   3.11667  5.5
```
"""
struct WeightedSubstitute{F<:Function, W<:AbstractArray{<:Real}} <: Imputor
    statistic::F
    weights::W
end

function WeightedSubstitute(; statistic=defaultstats, weights)
    return WeightedSubstitute(statistic, weights)
end

function _impute!(data::AbstractArray{Union{T, Missing}}, imp::WeightedSubstitute) where T
    mask = .!ismissing.(data)
    x = imp.statistic(disallowmissing(data[mask]), imp.weights[mask])
    return Base.replace!(data, missing => x)
end

# Auxiliary functions defining our default substitution rules

@doc """
    defaultstats(data[, wv])

A set of default substitution rules using either `median` or `mode` based on the `eltype` of
the input `data`. Specific rules are summarized as follows.

* `Bool` elements use `mode`
* `Real` elements use `median`
* `Integer` elements where `nunique(data) / length(data) < 0.25` use `mode`
  (ratings, categorical codings, etc)
* `Integer` elements with mostly unique values use `median`
* `!Number` (non-numeric) elements use `mode` as the safest fallback
""" defaultstats

# If we're operating over Bools then we're probably better off using mode
defaultstats(data::AbstractArray{<:Bool}, args...) = _mode(data, args...)

# If we're operating over Reals then we should probably use the median
defaultstats(data::AbstractArray{<:Real}, args...) = median(data, args...)

# If we're operating over integers with relatively few unique values then we're
# likely dealing with either counts or a categorical coding, so mode is probably
# safer
function defaultstats(data::AbstractArray{T}, args...) where T <: Integer
    threshold = 0.25 * length(data)
    nunique = length(unique(data))
    if nunique < threshold
        return _mode(data, args...)
    else
        return round(T, median(data, args...))
    end
end

# Fallback to mode as many types won't support mean or median anyways
defaultstats(data::AbstractArray, args...) = _mode(data, args...)
