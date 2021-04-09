function trycopy(data)
    # Not all objects support `copy`, but we should use it to improve
    # performance if possible.
    try
        copy(data)
    catch
        deepcopy(data)
    end
end

function dim(data, d)
    # Special case d === nothing as this currently signifies the default colwise
    # operations that are being deprecated.
    if d === nothing
        Base.depwarn(
            "Imputing on matrices will require specifying `dims=2` or `dims=:cols` in a " *
            "future release, to maintain the current behaviour.",
            :dim
        )
        return 2
    # Special case tables and matrices using the `:rows` and `:cols` dims values
    elseif d in (:rows, :cols) && (istable(data) || isa(data, AbstractMatrix))
        return NamedDims.dim((:rows, :cols), d)
    # Fallback to whatever NameDims gives us
    else
        return NamedDims.dim(NamedDims.dimnames(data), d)
    end
end

# Remove this once the corresponding statsbase pull request is merged and tagged.
# https://github.com/JuliaStats/StatsBase.jl/pull/611
_mode(a::AbstractArray) = mode(a)

function _mode(a::AbstractVector, wv::AbstractArray{T}) where T <: Real
    isempty(a) && throw(ArgumentError("mode is not defined for empty collections"))
    length(a) == length(wv) || throw(ArgumentError(
        "data and weight vectors must be the same size, got $(length(a)) and $(length(wv))"
    ))

    # Iterate through the data
    mv = first(a)
    mw = first(wv)
    weights = Dict{eltype(a), T}()
    for (x, w) in zip(a, wv)
        _w = get!(weights, x, zero(T)) + w
        if _w > mw
            mv = x
            mw = _w
        end
        weights[x] = _w
    end

    return mv
end

function _check_index(index_values, len)
    index_length = length(index_values)
    if index_length != len
        throw(DimensionMismatch(
            "Length of index_values ($index_length) must match length of data ($len)."
        ))
    end

    !issorted(index_values) && throw(ArgumentError("index_values must be sorted."))
end
