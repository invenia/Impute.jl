"""
    DropObs()

Removes missing observations from the `AbstractArray` or `Tables.table` provided.

# Example
```jldoctest
julia> using Impute: DropObs, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, DropObs(); dims=2)
2×3 Array{Union{Missing, Float64},2}:
 1.0  2.0  5.0
 1.1  2.2  5.5
```
"""
struct DropObs <: Imputor end

# Special case impute! for vectors because we know filter! will work
impute!(data::Vector, imp::DropObs) = filter!(!ismissing, data)

function impute!(data::Vector{<:NamedTuple}, imp::DropObs)
    return filter!(r -> all(!ismissing, propertyvalues(r)), data)
end

impute(data::AbstractVector, imp::DropObs) = filter(!ismissing, data)

function impute(data::Vector{<:NamedTuple}, imp::DropObs)
    return filter(r -> all(!ismissing, propertyvalues(r)), data)
end

function impute(data::AbstractMatrix{Union{T, Missing}}, imp::DropObs; dims=1) where T
    return filterobs(obs -> all(!ismissing, obs), data; dims=dims)
end

function impute(table, imp::DropObs)
    @assert istable(table)
    rows = Tables.rows(table)

    # Unfortunately, we'll need to construct a new table
    # since Tables.rows is just an iterator
    filtered = Iterators.filter(rows) do r
        all(!ismissing, propertyvalues(r))
    end

    table = materializer(table)(filtered)
    return table
end


"""
    DropVars()


Finds variables with too many missing values in a `AbstractMatrix` or `Tables.table` and
removes them from the input data.

# Examples
```jldoctest
julia> using Impute: DropVars, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, DropVars(); dims=2)
1×5 Array{Union{Missing, Float64},2}:
 1.1  2.2  3.3  missing  5.5
```
"""
struct DropVars <: Imputor end

function impute!(data::Vector{<:NamedTuple}, imp::DropVars)
    return materializer(data)(impute(Tables.columns(data), imp))
end

function impute(data::AbstractMatrix{Union{T, Missing}}, imp::DropVars; dims=1) where T
    return filtervars(data; dims=dims) do vars
        all(!ismissing, vars)
    end
end

function impute(table, imp::DropVars)
    istable(table) || throw(MethodError(impute!, (table, imp)))
    cols = Tables.columns(table)

    cnames = Iterators.filter(propertynames(cols)) do cname
        all(!ismissing, getproperty(cols, cname))
    end

    selected = TableOperations.select(table, cnames...)
    table = materializer(table)(selected)
    return table
end

# Add impute! methods to override the default behaviour in imputors.jl
function impute!(data::AbstractMatrix{Union{T, Missing}}, imp::Union{DropObs, DropVars}) where T
    data = impute(data, imp)
    return data
end

impute!(data, imp::Union{DropObs, DropVars}) = impute(data, imp)
