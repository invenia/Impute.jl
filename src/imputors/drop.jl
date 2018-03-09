"""
    Drop <: Imputor

Removes missing values from the `AbstractArray` or `DataFrame` provided.
"""
type Drop <: Imputor end

"""
    impute!{T<:Any}(imp::Drop, ctx::Context, data::AbstractArray{T, 1})

Uses `filter!` to remove missing elements from the array.

# Arguments
* `imp::Drop`: this `Imputor` method
* `ctx::Context`: contextual information for missing data
* `data::AbstractArray{T, 1}`: the data to impute

# Returns
* `AbstractArray{T, 1}`: our data array with missing elements removed
"""
function impute!{T<:Any}(imp::Drop, ctx::Context, data::AbstractArray{T, 1})
    return filter!(x -> !ismissing(ctx, x), data)
end

"""
    impute!{T<:Any}(imp::Drop, ctx::Context, data::AbstractArray{T, 2})

Finds the missing rows in the matrix and uses a mask (Array{Bool, 1}) to return the
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
* `data::AbstractArray{T, 2}`: the data to impute

# Returns
* `AbstractArray{T, 2}`: a new matrix with missing rows removed
"""
function impute!{T<:Any}(imp::Drop, ctx::Context, data::AbstractArray{T, 2})
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
    m.deleterows!(data, find(map(r -> ismissing(ctx, r), m.eachrow(data))))
    return data
end
