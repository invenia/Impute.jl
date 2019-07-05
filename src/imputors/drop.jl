"""
    Drop <: Imputor

Removes missing values from the `AbstractArray` or `Tables.table` provided.

# Fields
* `context::AbstractContext`: A context which keeps track of missing data
  summary information
"""
struct Drop <: Imputor
    context::AbstractContext
end

"""Drop(; context=Context()) -> Drop"""
Drop(; context=Context()) = Drop(context)

"""
    impute!(imp::Drop, data::AbstractVector)

Uses `filter!` to remove missing elements from the array.

# Arguments
* `imp::Drop`: this `Imputor` method
* `data::AbstractVector`: the data to impute

# Returns
* `AbstractVector`: our data array with missing elements removed
"""
function impute!(imp::Drop, data::AbstractVector)
    imp.context() do c
        filter!(x -> !ismissing(c, x), data)
    end
end

"""
    impute!(imp::Drop, data::AbstractMatrix)

Finds the missing rows in the matrix and uses a mask (Vector{Bool}) to return the
`data` with those rows removed. Unfortunately, the mask approach requires copying the matrix.

NOTES (or premature optimizations):
* We use `view`, but this will change the type of the `data` by returning a `SubArray`
* We might be able to do something clever by:
    1. reshaping the data to a vector
    2. running `deleteat!` for the appropriate indices and
    3. reshaping the data back to the desired shape.

# Arguments
* `imp::Drop`: this `Imputor` method
* `data::AbstractMatrix`: the data to impute

# Returns
* `AbstractMatrix`: a new matrix with missing rows removed
"""
function impute!(imp::Drop, data::AbstractMatrix)
    imp.context() do c
        mask = map(i -> !ismissing(c, data[i, :]), 1:size(data, 1))
        return data[mask, :]
    end
end

"""
    impute!(imp::Drop, table)

Finds the missing rows in the table and deletes them.

# Arguments
* `imp::Drop`: this `Imputor` method
* `table`: a type that implements the Tables API.

# Returns
* our data with the missing rows removed.
"""
function impute!(imp::Drop, table)
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
