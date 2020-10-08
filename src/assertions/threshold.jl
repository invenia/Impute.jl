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
    if t.weights === nothing
        mratio = count(ismissing, data) / length(data)
        if mratio > t.ratio
            throw(AssertionError("Ratio of missing values exceeded $(t.ratio) ($mratio)."))
        end
    else
        if size(data) != size(t.weights)
            throw(DimensionMismatch(string(
                "Input has dimensions $(size(data)), but thresholds weights ",
                "has dimensions $(size(t.weights))"
            )))
        end

        mratio = sum(t.weights[ismissing.(data)]) / sum(t.weights)
        if mratio > t.ratio
            throw(AssertionError(
                "Weighted ratio of missing values exceeded $(t.ratio) ($mratio)."
            ))
        end
    end

    return data
end
