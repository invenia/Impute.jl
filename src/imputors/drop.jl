struct DropObs <: Imputor
    vardim::Int
    context::AbstractContext
end

"""
    DropObs(; vardim=2, context=Context)

Removes missing observations from the `AbstractArray` or `Tables.table` provided.

# Keyword Arguments
* `vardim=2::Int`: Specify the dimension for variables in matrix input data
* `context::AbstractContext=Context()`: A context which keeps track of missing data
  summary information

# Example
```jldoctest
julia> using Impute: DropObs, Context, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, DropObs(; vardim=1, context=Context(; limit=1.0)))
2×3 Array{Union{Missing, Float64},2}:
 1.0  2.0  5.0
 1.1  2.2  5.5
```
"""
DropObs(; vardim=2, context=Context()) = DropObs(vardim, context)

function impute!(data::AbstractVector, imp::DropObs)
    imp.context() do c
        filter!(x -> !ismissing(c, x), data)
    end
end

function impute!(data::AbstractMatrix, imp::DropObs)
    imp.context() do c
        return filterobs(imp, data) do obs
            !ismissing(c, obs)
        end
    end
end

# Deleting elements from subarrays doesn't work so we need to collect that data into
# a separate array.
impute!(data::SubArray, imp::DropObs) = impute!(collect(data), imp::DropObs)

function impute!(table, imp::DropObs)
    imp.context() do c
        @assert istable(table)
        rows = Tables.rows(table)

        # Unfortunately, we'll need to construct a new table
        # since Tables.rows is just an iterator
        table = Iterators.filter(rows) do r
            !any(x -> ismissing(c, x), propertyvalues(r))
        end |> materializer(table)

        return table
    end
end


struct DropVars <: Imputor
    vardim::Int
    context::AbstractContext
end

"""
    DropVars(; vardim=2, context=Context())


Finds variables with too many missing values in a `AbstractMatrix` or `Tables.table` and
removes them from the input data.

# Keyword Arguments
* `vardim=2::Int`: Specify the dimension for variables in matrix input data
* `context::AbstractContext`: A context which keeps track of missing data
  summary information

# Examples
```jldoctest
julia> using Impute: DropVars, Context, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, DropVars(; vardim=1, context=Context(; limit=0.2)))
1×5 Array{Union{Missing, Float64},2}:
 1.1  2.2  3.3  missing  5.5
```
"""
DropVars(; vardim=2, context=Context()) = DropVars(vardim, context)

function impute!(data::AbstractMatrix, imp::DropVars)
    return filtervars(imp, data) do var
        try
            imp.context() do c
                for x in var
                    ismissing(c, x)
                end
            end
            return true
        catch e
            if isa(e, ImputeError)
                return false
            else
                rethrow(e)
            end
        end
    end
end

function impute!(table, imp::DropVars)
    @assert istable(table)
    cols = Tables.columns(table)

    cnames = Iterators.filter(propertynames(cols)) do cname
        try
            imp.context() do c
                col = getproperty(cols, cname)
                for i in eachindex(col)
                    ismissing(c, col[i])
                end
            end
            return true
        catch e
            if isa(e, ImputeError)
                return false
            else
                rethrow(e)
            end
        end
    end

    table = Tables.select(table, cnames...) |> materializer(table)
    return table
end
