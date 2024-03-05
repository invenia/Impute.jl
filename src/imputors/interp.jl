"""
    Interpolate(; limit=nothing)

Performs linear interpolation between the nearest values in an vector.
The current implementation is univariate, so each variable in a table or matrix will
be handled independently.

!!! Missing values at the head or tail of the array cannot be interpolated if there
are no existing values on both sides. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments
* `limit::Union{UInt, Nothing}`: Optionally limit the gap sizes that can be interpolated.

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
end

Interpolate(; limit=nothing) = Interpolate(limit)

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
                    diff = data[next_idx] - data[prev_idx]
                    incr = diff / T(gap_sz + 1)
                    val = data[prev_idx] + incr

                    # Iteratively fill in the values
                    for j in i:(next_idx - 1)
                        data[j] = val
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

function _impute!(data::AbstractVector{<:Union{T, Missing}}, imp::Interpolate) where {T<:Union{Signed, Unsigned}}
    dataf = _impute!(float(data), imp)
    data .= round.(Union{T, Missing}, dataf)
    return data
end
