"""
    NOCB()

Next observation carried backward (NOCB) iterates backwards through the `data` and fills
missing data with the next existing observation.

See also:
- [LOCF](@ref): Last Observation Carried Forward

WARNING: missing elements at the tail of the array may not be imputed if there is no
existing observation to carry backward. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments

# Example
```jldoctest
julia> using Impute: NOCB, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, NOCB(); dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  5.0  5.0  5.0
 1.1  2.2  3.3  5.5  5.5
```
"""
struct NOCB <: Imputor end

function _impute!(data::AbstractVector{Union{T, Missing}}, imp::NOCB) where T
    @assert !all(ismissing, data)
    end_idx = findlast(!ismissing, data) - 1

    for i in end_idx:-1:firstindex(data)
        if ismissing(data[i])
            data[i] = data[i+1]
        end
    end

    return data
end
