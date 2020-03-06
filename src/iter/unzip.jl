struct Unzip{I}
    xs::I
    i::Int
end

# TODO: Use parent eltype?

function unzip(xs::I) where {I}
    head = @ifsomething something(peek(peekiter(xs)))
    return Tuple(Unzip{I}(xs, i) for i in eachindex(head))
end

function unzip(f::Function, xs::I) where {I}
    head = @ifsomething something(peek(peekiter(xs)))
    return Tuple(f(Unzip{I}(xs, i)) for i in eachindex(head))
end

function iterate(it::Unzip{I}, state=nothing)
    val, st = if state === nothing
        @ifsomething(iterate(it.xs))
    else
        @ifsomething(iterate(it.xs, state))
    end

    return (val[it.i], st)
end

# unzip

# Special case where the input is a vector of tuples (e.g., map over zip)
# NOTE: This is relatively efficient for the map context, but is less efficient if you
# simply want to pass an iterator (e.g., Zip) directly.
function unzip(vals::Vector{T}) where T <: Tuple
    isempty(vals) && return tuple()
    n = length(vals)

    # Preallocate our vectors
    results = Tuple(Vector{t}(undef, n) for t in fieldtypes(T))

    for i in eachindex(vals)
        t = vals[i]

        for j in eachindex(t)
            results[j][i] = t[j]
        end
    end

    return results
end
