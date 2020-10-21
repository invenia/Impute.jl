"""
    ThresholdError <: Exception

Is thrown when a Threshold limit is exceed.

# Fields
* limit::Float64 - the threshold limit.
* value::Float64 - the missing data ratio identified
"""
struct ThresholdError <: Exception
    limit::Float64
    value::Float64
end

function Base.showerror(io::IO, err::ThresholdError)
    println(
        io,
        "ThresholdError: Missing data limit exceeded $(err.limit) ($(err.value))",
    )
end

"""
    Threshold(; limit=0.1)

Assert that the ratio of missing values in the provided dataset does not exceed to
specified limit.

# Keyword Arguments
* `limit::Real`: Allowed proportion of missing values (should be between 0.0 and 1.0).
"""
struct Threshold <: Validator
    limit::Float64
    weights::Union{AbstractWeights, Nothing}
end

Threshold(; limit=0.1, weights=nothing) = Threshold(limit, weights)

function _validate(data::AbstractArray{Union{T, Missing}}, t::Threshold) where T
    mratio = count(ismissing, data) / length(data)
    mratio > t.limit && throw(ThresholdError(t.limit, mratio))
    return data
end

"""
    WeightedThreshold(; limit, weights)

Assert that the weighted proportion missing values in the provided dataset does not exceed
to specified limit. The weighed proportion is calculated as
`sum(weights[ismissing.(data)]) / sum(weights)`

# Keyword Arguments
* `limit::Real`: Allowed proportion of missing values (should be between 0.0 and 1.0).
* `weights::AbstractWeights`: A set of statistical weights to use when evaluating the importance
  of each observation.
"""
struct WeightedThreshold{W <: AbstractArray{<:Real}} <: Validator
    limit::Float64
    weights::W
end

WeightedThreshold(; limit, weights) = WeightedThreshold(limit, weights)

function _validate(data::AbstractArray{Union{T, Missing}}, wt::WeightedThreshold) where T
    if size(data) != size(wt.weights)
        throw(DimensionMismatch(string(
            "Input has dimensions $(size(data)), but thresholds weights ",
            "has dimensions $(size(wt.weights))"
        )))
    end

    val = sum(wt.weights[ismissing.(data)]) / sum(wt.weights)
    val > wt.limit && throw(ThresholdError(wt.limit, val))

    return data
end
