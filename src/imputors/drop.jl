"""
    DropObs <: Imputor

Removes missing values from the `AbstractArray` or `Tables.table` provided.

# Fields
* `context::AbstractContext`: A context which keeps track of missing data
  summary information
"""
struct DropObs <: Imputor
    vardim::Int
    context::AbstractContext
end

DropObs(; vardim=2, context=Context()) = DropObs(vardim, context)

"""
    impute!(imp::DropObs, data::AbstractVector)

Uses `filter!` to remove missing elements from the array.

# Arguments
* `imp::DropObs`: this `Imputor` method
* `data::AbstractVector`: the data to impute

# Returns
* `AbstractVector`: our data array with missing elements removed
"""
function impute!(imp::DropObs, data::AbstractVector)
    imp.context() do c
        filter!(x -> !ismissing(c, x), data)
    end
end

"""
    impute!(imp::DropObs, data::AbstractMatrix)

Finds the missing rows in the matrix and uses a mask (Vector{Bool}) to return the
`data` with those rows removed. Unfortunately, the mask approach requires copying the matrix.

NOTES (or premature optimizations):
* We use `view`, but this will change the type of the `data` by returning a `SubArray`
* We might be able to do something clever by:
    1. reshaping the data to a vector
    2. running `deleteat!` for the appropriate indices and
    3. reshaping the data back to the desired shape.

# Arguments
* `imp::DropObs`: this `Imputor` method
* `data::AbstractMatrix`: the data to impute

# Returns
* `AbstractMatrix`: a new matrix with missing rows removed
"""
function impute!(imp::DropObs, data::AbstractMatrix)
    imp.context() do c
        return filterobs(imp, data) do obs
            !ismissing(c, obs)
        end
    end
end

"""
    impute!(imp::DropObs, table)

Finds the missing rows in the table and deletes them.

# Arguments
* `imp::DropObs`: this `Imputor` method
* `table`: a type that implements the Tables API.

# Returns
* our data with the missing rows removed.
"""
function impute!(imp::DropObs, table)
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


"""
    DropVars <: Imputor


Removes missing values from the `AbstractArray` or `Tables.table` provided.

# Fields
* `context::AbstractContext`: A context which keeps track of missing data
  summary information
"""
struct DropVars <: Imputor
    vardim::Int
    context::AbstractContext
end

DropVars(; vardim=2, context=Context()) = DropVars(vardim, context)

"""
    impute!(imp::DropVars, data::AbstractMatrix)

Finds columns in the matrix with too many missing values and uses a mask (Vector{Bool}) to
return the `data` with those columns removed. Unfortunately, the mask approach
requires copying the matrix.

# Arguments
* `imp::DropVars`: this `Imputor` method
* `data::AbstractMatrix`: the data to impute

# Returns
* `AbstractMatrix`: a new matrix with missing columns removed
"""
function impute!(imp::DropVars, data::AbstractMatrix)
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

"""
    impute!(imp::DropVars, table)

Find remove columns in the table with too many missing elements.

# Arguments
* `imp::DropVars`: this `Imputor` method
* `table`: a type that implements the Tables API.

# Returns
* our data with the missing columns removed.
"""
function impute!(imp::DropVars, table)
    @assert istable(table)
    cols = Tables.columns(table)

    cnames = Iterators.filter(propertynames(cols)) do cname
        try
            imp.context() do c
                col = getproperty(cols, cname)
                for i in 1:length(col)
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
