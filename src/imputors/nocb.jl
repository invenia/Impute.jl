"""
    NOCB(; limit=nothing)

Next observation carried backward (NOCB) iterates backwards through the `data` and fills
missing data with the next existing observation.

See also:
- [`Impute.LOCF`](@ref): Last Observation Carried Forward

!!! Missing elements at the tail of the array may not be imputed if there is no
existing observation to carry backward. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments
* `limit::Union{UInt, Nothing}`: Optionally limits the amount of consecutive missing values to replace.

# Example
```jldoctest
julia> using Impute: NOCB, impute

julia> M = [1.0 2.0 missing missing missing 6.0; 1.1 missing missing 4.4 5.5 6.6]
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0       missing   missing   missing  6.0
 1.1   missing  missing  4.4       5.5       6.6

julia> impute(M, NOCB(); dims=:rows)
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0  6.0  6.0  6.0  6.0
 1.1  4.4  4.4  4.4  5.5  6.6

julia> impute(M,  NOCB(; limit=2); dims=:rows)
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing  6.0  6.0  6.0
 1.1  4.4  4.4       4.4  5.5  6.6
```
"""
struct NOCB <: Imputor
    limit::Union{UInt, Nothing}
end

NOCB(; limit=nothing) = NOCB(limit)

function _impute!(data::AbstractVector{Union{T, Missing}}, imp::NOCB) where T
    @assert !all(ismissing, data)
    end_idx = findlast(!ismissing, data)
    count = 1

    for i in end_idx - 1:-1:firstindex(data)
        if ismissing(data[i])
            if imp.limit === nothing
                data[i] = data[i+1]
            elseif count <= imp.limit
                data[i] = data[end_idx]
                count += 1
            end
        else
            end_idx = i
            count = 1
        end
    end

    return data
end
