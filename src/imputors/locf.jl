"""
    LOCF())

Last observation carried forward (LOCF) iterates forwards through the `data` and fills
missing data with the last existing observation. The current implementation is univariate,
so each variable in a table or matrix will be handled independently.

See also:
- [NOCB](@ref): Next Observation Carried Backward

WARNING: missing elements at the head of the array may not be imputed if there is no
existing observation to carry forward. As a result, this method does not guarantee
that all missing values will be imputed.

# Example
```jldoctest
julia> using Impute: LOCF, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, LOCF(); dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.0  2.0  5.0
 1.1  2.2  3.3  3.3  5.5
```
"""
struct LOCF <: Imputor end

function _impute!(data::AbstractVector{Union{T, Missing}}, imp::LOCF) where T
    start_idx = findfirst(!ismissing, data)
    if start_idx === nothing
        @debug "Cannot carry forward points when all values are missing"
        return data
    end

    start_idx += 1
    for i in start_idx:lastindex(data)
        if ismissing(data[i])
            data[i] = data[i-1]
        end
    end

    return data
end
