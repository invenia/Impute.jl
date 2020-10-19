"""
    DeclareMissings(; values)

DeclareMissings (or replace) various missing data values with `missing`.
This is useful for downstream imputation methods that assume missing data is represented by
a `missing`.

!!! In-place methods are only applicable for datasets which already `allowmissing`.

# Keyword Arguments
* `value::Tuple`: A tuple of values that should be considered `missing`

# Example
```jldoctest
julia> using Impute: DeclareMissings, apply

julia> M = [1.0 2.0 -9999.0 NaN 5.0; 1.1 2.2 3.3 0.0 5.5]
2×5 Array{Float64,2}:
 1.0  2.0  -9999.0  NaN    5.0
 1.1  2.2      3.3    0.0  5.5

julia> apply(M, DeclareMissings(; values=(NaN, -9999.0, 0.0)))
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5
```
"""
struct DeclareMissings{T<:Tuple}
    values::T
end

function DeclareMissings(; values)
    T = isa(values, Tuple) ? values : tuple(values)
    return DeclareMissings{typeof(T)}(T)
end

apply!(data::AbstractArray{Missing}, imp::DeclareMissings) = data

# Primary definition just calls `replace!`
function apply!(data::AbstractArray{Union{T, Missing}}, imp::DeclareMissings) where T
    # Reduce the possible set of values to those that could actually be found in the data
    # Useful, if we declare a `Replace` imputor that should be applied to multiple datasets.
    Base.replace!(data, (v => missing for v in imp.values if v isa T)...)
end

# Most of the time the in-place methods won't work because we need to change the
# eltype with allowmissing
apply(data::AbstractArray, imp::DeclareMissings) = apply!(allowmissing(data), imp)

# Custom implementation of a non-mutating impute for tables
function apply(table, imp::DeclareMissings)
    istable(table) || throw(MethodError(apply, (table, imp)))

    ctable = Tables.columns(table)

    cnames = Tuple(propertynames(ctable))
    cdata = Tuple(apply(getproperty(ctable, cname), imp) for cname in cnames)
    # Reconstruct as a ColumnTable
    result = NamedTuple{cnames}(cdata)

    # If our input was a ColumnTable just return the result. We can also do the same for
    if isa(table, Tables.ColumnTable)
        return result
    else
        return materializer(table)(result)
    end
end

# Specialcase for rowtable
function apply(data::T, imp::DeclareMissings) where T <: AbstractVector{<:NamedTuple}
    # We use columntable here so that we don't call `materialize` more often than needed.
    return materializer(data)(apply(Tables.columntable(data), imp))
end
