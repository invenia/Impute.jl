"""
    Drop <: Imputor

Removes missing values from the `AbstractArray` or `DataFrame` provided.
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
    ctx.num = size(data, 1)
    mask = map(i -> !ismissing(ctx, data[i, :]), 1:size(data, 1))
    return data[mask, :]
end

"""
    impute!(imp::Drop, ctx::Context, data::DataFrame)

Finds the missing rows in the `DataFrame` and deletes them.

NOTE: this isn't quite as fast as `dropnull` in `DataFrames`s as we're using an arbitrary
`missing` function rather than using the precomputed `dt.isnull` vector of bools.

# Arguments
* `imp::Drop`: this `Imputor` method
* `ctx::Context`: contextual information for missing data
* `data::DataFrame`: the data to impute

# Returns
* `DataFrame`: our data with the missing rows removed.
"""
function impute!(imp::Drop, ctx::Context, data::DataFrame)
    ctx.num = size(data, 1)
    m = typeof(data).name.module
    m.deleterows!(data, findall(r -> ismissing(ctx, r), m.eachrow(data)))
    return data
end
