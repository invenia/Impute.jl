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

# Example
```jldoctest
julia> using Impute: LOCF, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, LOCF(); dims=:rows)
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0  2.0  2.0  5.0
 1.1  2.2  3.3  3.3  5.5
```
"""
struct LOCF <: Imputor end

function _impute!(data::AbstractVector{Union{T, Missing}}, imp::LOCF) where T
    @assert !all(ismissing, data)
    start_idx = findfirst(!ismissing, data) + 1

    for i in start_idx:lastindex(data)
        if ismissing(data[i])
            data[i] = data[i-1]
        end
    end

    return data
end

"""
    LimitedLOCF(max_gap_size, [gap_axis::AbstractVector])

A limited version of last observation carried forward (LOCF).
Fills missing data only if the size of the gap is less than `max_gap_size`.

A sorted `gap_axis` can be applied to measure the gap by other metrics (eg: timestamps).
The `gap_axis` must match the length of the vector being imputed.

See:
- [LOCF](@ref): Last Observation Carried Forward

!!! Missing elements at the head of the array may not be imputed if there is no
existing observation to carry forward. As a result, this method does not guarantee
that all missing values will be imputed.

# Example
```jldoctest
julia> using Impute: LimitedLOCF, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 missing 3.3 4.4 5.5]
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0        missing   missing  5.0
 1.1   missing  3.3       4.4       5.5

julia> impute(M,  LimitedLOCF(1); dims=:rows)
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing   missing  5.0
 1.1  1.1  3.3       4.4       5.5

julia> distances = [0.1, 0.6, 0.7, 0.8, 0.9];

julia> impute(M,  LimitedLOCF(0.3, distances); dims=:rows)
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0       2.0  2.0  5.0
 1.1   missing  3.3  4.4  5.5
```
"""
struct LimitedLOCF <: Imputor
    max_gap_size
    gap_axis::Union{AbstractVector, Nothing}

    function LimitedLOCF(max_gap_size, gap_axis=nothing)
        gap_axis !== nothing && !issorted(gap_axis) && error("gap_axis must be sorted")
        return new(max_gap_size, gap_axis)
    end
end

function LimitedLOCF(; max_gap_size, gap_axis=nothing)
    return LimitedLOCF(max_gap_size, gap_axis)
end

function _impute!(data::AbstractVector{Union{T, Missing}}, imp::LimitedLOCF) where T
    gap_axis = imp.gap_axis
    n = length(data)

    if gap_axis === nothing
        gap_axis = Base.OneTo(n)
    elseif length(gap_axis) != n
        throw(DimensionMismatch(
            "Length of gap_axis ($(length(gap_axis))) must match length of data ($n)."
        ))
    end

    next_data = findfirst(!ismissing, data)

    while next_data !== nothing
        gap_start = findnext(ismissing, data, next_data)
        gap_start === nothing && break # No more missing data, exit early

        # Use the last non-missing value
        fill_value = data[gap_start - 1]

        # Find the end of the block of missings
        next_data = findnext(!ismissing, data, gap_start)
        gap_end = next_data === nothing ? n : next_data - 1

        # Replace values only if the time-gap is small enough
        if gap_axis[gap_end] - gap_axis[gap_start - 1] <= imp.max_gap_size
            data[gap_start:gap_end] .= fill_value
        end
    end

    return data
end
