"""
    Standardize(; values)

Standardize (or replace) various missing data values with `missing`.
This is useful for downstream imputation methods that assume missing data is represented by
a `missing`.

Warning: In-place methods are only applicable for datasets which already `allowmissing`.

# Keyword Arguments
* `value::Tuple`: A tuple of values that should be considered `missing`

# Example
```jldoctest
julia> using Impute: Standardize, impute

julia> M = [1.0 2.0 -9999.0 NaN 5.0; 1.1 2.2 3.3 0.0 5.5]
2×5 Array{Float64,2}:
 1.0  2.0  -9999.0  NaN    5.0
 1.1  2.2      3.3    0.0  5.5

julia> impute(M, Standardize(; values=(NaN, -9999.0, 0.0)))
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5
```
"""
struct Standardize <: Imputor
    values::Tuple
end

Standardize(; values::Tuple) = Standardize(values)

# Primary definition just calls `replace!`
function _impute!(data::AbstractArray{Union{T, Missing}}, imp::Standardize) where T
    # Reduce the possible set of values to those that could actually be found in the data
    # Useful, if we declare a `Replace` imputor that should be applied to multiple datasets.
    Base.replace!(
        data,
        (v => missing for v in Base.filter(v -> isa(v, T), imp.values))...
    )
end

# Most of the time the in-place methods won't work because we need to change the
# eltype with allowmissing
impute(data::AbstractArray, imp::Standardize) = _impute!(allowmissing(data), imp)

# Custom implementation of a non-mutating impute for tables
function impute(table, imp::Standardize)
    istable(table) || throw(MethodError(impute, (table, imp)))

    ctable = Tables.columns(table)

    cnames = Tuple(propertynames(ctable))
    cdata = Tuple(impute(getproperty(ctable, cname), imp) for cname in cnames)
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
function impute(data::T, imp::Standardize) where T <: AbstractVector{<:NamedTuple}
    # We use columntable here so that we don't call `materialize` more often than needed.
    return materializer(data)(impute(Tables.columntable(data), imp))
end

# Awkward imputor overrides necessary because we intercepted the higher level
# `impute` calls
_impute!(data::AbstractArray{Missing}, imp::Standardize) = data

# Skip custom dims stuff cause it isn't necessary here.
function impute!(data::AbstractMatrix{Union{T, Missing}}, imp::Standardize) where {T}
    return _impute!(data, imp)
end

# Just throw a method error for mutation of tables as the operation would
# otherwise be undefined depending on the table type.
# impute!(table, imp::Standardize) = throw(MethodError(impute!, (table, imp)))
