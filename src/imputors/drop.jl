"""
    Drop <: Imputor

Removes missing values from the `AbstractArray` or `Tables.table` provided.
"""
struct Drop <: Imputor end

"""
    impute!(imp::Drop, ctx::Context, data::AbstractVector)

Uses `filter!` to remove missing elements from the array.

# Arguments
* `imp::Drop`: this `Imputor` method
* `ctx::Context`: contextual information for missing data
* `data::AbstractVector`: the data to impute

# Returns
* `AbstractVector`: our data array with missing elements removed
"""
function impute!(imp::Drop, ctx::Context, data::AbstractVector)
    return filter!(x -> !ismissing(ctx, x), data)
end

"""
    impute!(imp::Drop, ctx::Context, data::AbstractMatrix)

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
* `ctx::Context`: contextual information for missing data
* `data::AbstractMatrix`: the data to impute

# Returns
* `AbstractMatrix`: a new matrix with missing rows removed
"""
function impute!(imp::Drop, ctx::Context, data::AbstractMatrix)
    mask = ctx() do c
        map(i -> !ismissing(ctx, data[i, :]), 1:size(data, 1))
    end

    return data[mask, :]
end

"""
    impute!(imp::Drop, ctx::Context, table)

Finds the missing rows in the table and deletes them.

# Arguments
* `imp::Drop`: this `Imputor` method
* `ctx::Context`: contextual information for missing data
* `table`: a type that implements the Tables API.

# Returns
* our data with the missing rows removed.
"""
function impute!(imp::Drop, ctx::Context, table)
    @assert istable(table)
    rows = Tables.rows(table)

    result = Iterators.filter(rows) do r
        !any(x -> ismissing(ctx, x), propertyvalues(r))
    end

    # Unfortunately, we'll need to construct a new table
    # since Tables.rows is just an iterator
    table = materializer(table)(result)
    return table
end
