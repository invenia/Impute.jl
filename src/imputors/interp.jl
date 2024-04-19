"""
    Interpolate(; limit=nothing, r=nothing)

Performs linear interpolation between the nearest values in an vector.
The current implementation is univariate, so each variable in a table or matrix will
be handled independently.

!!! Missing values at the head or tail of the array cannot be interpolated if there
are no existing values on both sides. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments
* `limit::Union{UInt, Nothing}`: Optionally limit the gap sizes that can be interpolated.
* `r::Union{RoundingMode, Nothing}`: Optionally specify a rounding mode.
    Avoids `InexactError`s when interpolating over integers.

# Example
```jldoctest
julia> using Impute: Interpolate, impute

julia> M = [1.0 2.0 missing missing missing 6.0; 1.1 missing missing 4.4 5.5 6.6]
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0       missing   missing   missing  6.0
 1.1   missing  missing  4.4       5.5       6.6

julia> impute(M, Interpolate(); dims=:rows)
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0  3.0  4.0  5.0  6.0
 1.1  2.2  3.3  4.4  5.5  6.6

julia> impute(M, Interpolate(; limit=2); dims=:rows)
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing   missing   missing  6.0
 1.1  2.2  3.3       4.4       5.5       6.6
```
"""
struct Interpolate <: Imputor
    limit::Union{UInt, Nothing}
    r::Union{RoundingMode, Nothing}
end

Interpolate(; limit=nothing, r=nothing) = Interpolate(limit, r)

function _impute!(data::AbstractVector{<:Union{T, Missing}}, imp::Interpolate) where T
    @assert !all(ismissing, data)
    i = findfirst(!ismissing, data) + 1

    while i < lastindex(data)
        if ismissing(data[i])
            j = _findnext(data, i + 1)

            if j !== nothing
                if imp.limit === nothing || j - i + 1 <= imp.limit
                    _interpolate!(data, i:j, data[i - 1], data[j + 1], imp.r)
                end

                i = j + 1
            else
                break
            end
        end
        i += 1
    end

    return data
end

# Our kernel function used to avoid type instability issues.
# https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions
function _interpolate!(data, indices, prev, next, r)
    incr = _calculate_increment(prev, next, length(indices) + 1)

    for (i, k) in enumerate(indices)
        data[k] = _calculate_value(prev, incr, i, r)
    end
end

# Utility function for finding the last index within a missing data block
function _findnext(data, i)
    j = findnext(!ismissing, data, i)
    j === nothing && return j
    return j - 1
end

# Calculates the increment for interpolation
_calculate_increment(a, b, n) = (b - a) / n
# Special case for avoiding integer overflow
_calculate_increment(a::T, b::T, n) where {T<:Unsigned} = _calculate_increment(Int(a), Int(b), n)

# Calculates the interpolated value for a given iteration i
# Default case of simply prev + incr * i
_calculate_value(prev, incr, i, r) = prev + incr * i
# Special case for rounding integers
_calculate_value(prev::T, incr, i, r::RoundingMode) where {T<:Integer} = round(T, prev + incr * i, r)
