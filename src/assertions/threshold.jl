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
        "ThresholdError: Ratio of missing values exceeded $(err.limit): $(err.value)",
    )
end

"""
    Threshold(; ratio=0.1, weights=nothing)

Assert that the ratio of missing values in the provided dataset does not exceed to specified ratio.
If a weights array is provided then the ratio will be calculated as the
`sum(weights[ismissing.(data)]) / sum(weights)`

# Keyword Arguments
* `ratio::Real`: Allowed proportion of missing values (should be between 0.0 and 1.0).
* `weights::AbstractWeights`: A set of statistical weights to use when evaluating the importance
  of each observation. If present a weighted ratio of missing values will be calculated.
"""
struct Threshold <: Assertion
    ratio::Float64
    weights::Union{AbstractWeights, Nothing}
end

Threshold(; ratio=0.1, weights=nothing) = Threshold(ratio, weights)

function _assert(data::AbstractArray{Union{T, Missing}}, t::Threshold) where T
    mratio = if t.weights === nothing
        count(ismissing, data) / length(data)
    else
        if size(data) != size(t.weights)
            throw(DimensionMismatch(string(
                "Input has dimensions $(size(data)), but thresholds weights ",
                "has dimensions $(size(t.weights))"
            )))
        end

        sum(t.weights[ismissing.(data)]) / sum(t.weights)
    end

    mratio > t.ratio && throw(ThresholdError(t.ratio, mratio))

    return data
end
