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
