"""
    Drop <: Imputor

Removes missing values from the `AbstractArray` or `Tables.table` provided.
"""
struct Drop <: Imputor end

"""
    impute!(imp::Drop, context::AbstractContext, data::AbstractVector)

Uses `filter!` to remove missing elements from the array.

# Arguments
* `imp::Drop`: this `Imputor` method
* `context::AbstractContext`: contextual information for missing data
* `data::AbstractVector`: the data to impute

# Returns
* `AbstractVector`: our data array with missing elements removed
"""
function impute!(imp::Drop, context::AbstractContext, data::AbstractVector)
    context() do c
        filter!(x -> !ismissing(c, x), data)
    end
end

"""
    impute!(imp::Drop, context::AbstractContext, data::AbstractMatrix)

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
* `context::AbstractContext`: contextual information for missing data
* `data::AbstractMatrix`: the data to impute

# Returns
* `AbstractMatrix`: a new matrix with missing rows removed
"""
function impute!(imp::Drop, context::AbstractContext, data::AbstractMatrix)
    context() do c
        mask = map(i -> !ismissing(c, data[i, :]), 1:size(data, 1))
        return data[mask, :]
    end
end

"""
    impute!(imp::Drop, context::AbstractContext, table)

Finds the missing rows in the table and deletes them.

# Arguments
* `imp::Drop`: this `Imputor` method
* `context::AbstractContext`: contextual information for missing data
* `table`: a type that implements the Tables API.

# Returns
* our data with the missing rows removed.
"""
function impute!(imp::Drop, context::AbstractContext, table)
    context() do c
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
