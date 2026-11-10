"""
    Extrapolate(; limit=nothing, r=nothing)

Performs linear extrapolation at the boundaries of the data assuming a linear relationship.

See also:
- [`Impute.Interpolate`]
- [`Impute.LOCF`]
- [`Impute.NOCB`]
"""
struct Extrapolate <: Imputor
    limit::Union{UInt, Nothing}
    r::Union{RoundingMode, Nothing}
end

Extrapolate(; limit=nothing, r=nothing) = Extrapolate(limit, r)

function _impute!(data::AbstractVector{Union{T, Missing}}, imp::Extrapolate) where T
    @assert !all(ismissing, data)
    _head!(data, imp)
    _tail!(data, imp)
    return data
end

function _head!(data::AbstractVector{Union{T, Missing}}, imp::Extrapolate) where T
    # Process head
    i = findfirst(!ismissing, data)
    @assert i !== nothing "Data must contain at least two non-missing value."
    j = findnext(!ismissing, data, nextind(data, i))
    @assert j !== nothing "Data must contain at least two non-missing values."

    delta = _slope(i, data[i], j, data[j])
    idx = firstindex(data):prevind(data, i)

    # Early exit if the leading gap exceeds the limit
    imp.limit !== nothing && length(idx) > imp.limit && return data
    foreach(k -> data[k] = _calc(data[i], delta, k - i, imp.r), idx)

    return data
end

function _tail!(data::AbstractVector{Union{T, Missing}}, imp::Extrapolate) where T
    # Process tail
    j = findlast(!ismissing, data)
    @assert j !== nothing "Data must contain at least two non-missing value."
    i = findprev(!ismissing, data, prevind(data, j))
    @assert i !== nothing "Data must contain at least two non-missing values."

    delta = _slope(i, data[i], j, data[j])
    idx = nextind(data, j):lastindex(data)

    # Early exit if the leading gap exceeds the limit
    imp.limit !== nothing && length(idx) > imp.limit && return data
    foreach(k -> data[k] = _calc(data[i], delta, k - i, imp.r), idx)

    return data
end

function _slope(i, x::T, j, y::T) where T
    @assert i < j
    return T <: Unsigned ? (Int(y) - Int(x)) / (j - i) : (y - x) / (j - i)
end

_calc(nearest, slope, offset, r) = nearest + slope * offset
_calc(nearest::T, slope, offset, r::RoundingMode) where {T<:Integer} = nearest + round(slope * offset, r)
