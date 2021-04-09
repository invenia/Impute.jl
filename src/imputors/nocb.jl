"""
    NOCB()

Next observation carried backward (NOCB) iterates backwards through the `data` and fills
missing data with the next existing observation.

See also:
- [`Impute.LOCF`](@ref): Last Observation Carried Forward

!!! Missing elements at the tail of the array may not be imputed if there is no
existing observation to carry backward. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments
The following additional keyword arguments can be applied during impute:
* `limit=Inf`: Limits the amount of consecutive missing values to replace.
* `index_values::AbstractVector`: A sorted vector of index information for the data.
  This is compared when using `limit` to limit replacement of data at irregular intervals
  or by other types (eg: timestamps).

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

julia> impute(M,  NOCB(); dims=:rows, limit=2)
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing  6.0  6.0  6.0
 1.1  4.4  4.4       4.4  5.5  6.6

julia> impute(M,  NOCB(); dims=:rows, limit=0.5, index_values=[0.1, 0.2, 0.7, 0.8, 0.9, 1.3])
2×6 Matrix{Union{Missing, Float64}}:
 1.0  2.0        missing  6.0  6.0  6.0
 1.1   missing  4.4       4.4  5.5  6.6
```
"""
struct NOCB <: Imputor end

function _impute!(
    data::AbstractVector{Union{T, Missing}},
    imp::NOCB;
    limit=Inf,
    index_values::AbstractVector=Base.OneTo(length(data)),
) where T

    @assert !all(ismissing, data)
    _check_index(index_values, length(data))

    end_idx = findlast(!ismissing, data)

    for i in end_idx - 1:-1:firstindex(data)
        if ismissing(data[i])
            if index_values[end_idx] - index_values[i] <= limit
                data[i] = data[i+1]
            end
        else
            end_idx = i
        end
    end

    return data
end
