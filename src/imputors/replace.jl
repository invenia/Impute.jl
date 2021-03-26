"""
    Replace(; value)

Replace `missing`s with one of the specified constant values, depending on the input type.
If multiple values of the same type are provided then the first one will be used.
If the input data is of a different type then the no replacement will be performed.

# Keyword Arguments
* `values::Tuple`: A scalar or tuple of different values that should be used to replace
  missings. Typically, one value per type you're considering imputing for.

# Example
```jldoctest
julia> using Impute: Replace, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Replace(; values=0.0); dims=2)
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0  0.0  0.0  5.0
 1.1  2.2  3.3  0.0  5.5
```
"""
struct Replace <: Imputor
    values::Tuple
end

Replace(; values) = isa(values, Tuple) ? Replace(values) : Replace(tuple(values))

function _impute!(data::AbstractArray{Union{T, Missing}}, imp::Replace) where T
    i = findfirst(x -> isa(x, T), imp.values)
    i === nothing && return data
    return Base.replace!(data, missing => imp.values[i])
end
