"""
    LOCF()

Last observation carried forward (LOCF) iterates forwards through the `data` and fills
missing data with the last existing observation. The current implementation is univariate,
so each variable in a table or matrix will be handled independently.

See also:
- [`Impute.NOCB`](@ref): Next Observation Carried Backward

!!! Missing elements at the head of the array may not be imputed if there is no
existing observation to carry forward. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments
The following additional keyword arguments can be applied during impute:
* `limit=Inf`: Limits the amount of consecutive missing values to replace.
* `index_values::AbstractVector`: A sorted vector of index information for the data.
  This is compared when using `limit` to limit replacement of data at irregular intervals
  or by other types (eg: timestamps).

# Example
```jldoctest
julia> using Impute: LOCF, impute

julia> M = [1.0 2.0 missing missing missing 6.0; 1.1 missing missing 4.4 5.5 6.6]
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0       missing   missing   missing  6.0
 1.1   missing  missing  4.4       5.5       6.6

julia> impute(M, LOCF(); dims=:rows)
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0  2.0  2.0  2.0  6.0
 1.1  1.1  1.1  4.4  5.5  6.6

julia> impute(M,  LOCF(; limit=2); dims=:rows)
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0  2.0  2.0   missing  6.0
 1.1  1.1  1.1  4.4  5.5       6.6
```
"""
struct LOCF <: Imputor
    limit::Union{UInt, Nothing}
end

LOCF(; limit=nothing) = LOCF(limit)

function _impute!(data::AbstractVector{Union{T, Missing}}, imp::LOCF) where T
    @assert !all(ismissing, data)
    start_idx = findfirst(!ismissing, data)
    count = 1

    for i in start_idx + 1:lastindex(data)
        if ismissing(data[i])
            if imp.limit === nothing
                data[i] = data[i-1]
            elseif count <= imp.limit
                data[i] = data[start_idx]
                count += 1
            end
        else
            start_idx = i
            count = 1
        end
    end

    return data
end
