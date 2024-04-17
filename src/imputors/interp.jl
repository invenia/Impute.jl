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
                    inc = _calculate_increment(data[prev_idx], data[next_idx], gap_sz+1)
                    gen = _gen_interp(data[prev_idx], inc, gap_sz+1, imp.r)
                    _gen_set!(data, prev_idx, gen)
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

# sets vector slice via a generator (faster)
function _gen_set!(v::AbstractVector, after::Integer, gen)
    for (i, val) in enumerate(gen)
       v[after+i] = val
    end
end

# generator of interpolated values
_gen_interp(a, inc, n, r) = (a + inc*i for i=1:n)
_gen_interp(a::T, inc, n, r::RoundingMode) where {T<:Integer} = (round(T, a + inc*i, r) for i=1:n)

_calculate_increment(a, b, n) = (b - a) / n
_calculate_increment(a::T, b::T, n) where {T<:Unsigned} = _calculate_increment(Int(a), Int(b), n)
