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
            prev_idx = i - 1
            next_idx = findnext(!ismissing, data, i + 1)

            if next_idx !== nothing
                gap_sz = (next_idx - prev_idx) - 1

                if imp.limit === nothing || gap_sz <= imp.limit
                    prev = data[prev_idx]
                    next = data[next_idx]
                    incr = _calculate_increment(prev, next, gap_sz + 1, imp.r)
                    val = prev + incr

                    # Iteratively fill in the values
                    # Determine hi and lo values for clamping in the loop
                    # According to @benchmark calling extrema with a tuple has the same
                    # performance as calling min/max individually.
                    lo, hi = extrema((prev, next))
                    for j in i:(next_idx - 1)
                        data[j] = clamp(val, lo, hi)
                        val += incr
                    end
                end

                i = next_idx
            else
                break
            end
        end
        i += 1
    end

    return data
end

# Default cases where no rounding behaviour is specified
_calculate_increment(a, b, n, ::Nothing) = (b - a) / n
_calculate_increment(a::Unsigned, b::Unsigned, n, r::Nothing) = _calculate_increment(Int(a), Int(b), n, r)

# Pass a rounding mode to `div`
_calculate_increment(a, b, n, r) = div(b - a, n, r)
_calculate_increment(a::Unsigned, b::Unsigned, n, r) = _calculate_increment(Int(a), Int(b), n, r)
