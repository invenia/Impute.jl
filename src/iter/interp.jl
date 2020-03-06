struct Interpolate{I}
    xs::I
end

const StateType{T} = Tuple{Union{T, Missing}, Union{T, Missing, Nothing}, Int}

IteratorSize(::Type{<:Fill}) = HasLength()
IteratorEltype(::Type{Fill{I}}) where {I} = IteratorEltype(I)
eltype(::Type{Fill{I}}) where {I} = eltype(I)

function _iterate(it, state=nothing)
    t = state === nothing ? iterate(it.xs) : iterate(it.xs, tail(state))
    t === nothing && return nothing
    val, xs_state = Iterators.peel(t)

    # Starting case
    res = if state === nothing
        map(val) do x
            (x, (x, missing, 0))
        end
    else
        map(zip(val, first(state))) do (x, s)
            update(it, x, s, xs_state)
        end
    end


end

# Single missing value with empty state
update(it::Interpolate, val::Missing, ::Nothing, xs_st) = (val, (val, missing, 0))
# Non missing value with empty state
update(it::Interpolate, val, ::Nothing, xs) = (val, map(x -> (x, missing, 0), val))

function update(it::Interpolate, val, st, xs_st)
    n = length(val)

    # If length is 1 and the eltype and return type are the same then we assume it's a scalar
    if n == 1 && eltype(val) == typeof(val)
        return (val, (val, missing, 0))
    else
        i = 1  # Don't use `enumerate` as that can can change the return type from `map`.

        new_val = map(val) do x
            v, s = update(it, x, st[i], xs_st)
        end
update(it::Interpolate, val::Missing, st::Tuple{Missing, T, Int}, xs_st) where T = (val, st)
update(it::Interpolate, val::Missing, st::Tuple{T, Nothing, Int}) where T = (val, st)
function update(it::Interpolate, val::Missing, st::Tuple{T, Missing, Int}, xs_st) where T
    t = findnext(it.xs, xs_st)
    if t === nothing
        return (val, (first(st), nothing, 0))
    else
        return update(it, val, (first(st), t...), xs_st)
    end
end

function update(it::Interpolate, val::Missing, st::Tuple{T, T, Int}, xs_st) where T
    lo, no, n = st
    new_val = T(lo + (no - lo) / n)
    n -= 1

    if n < 1
        return (new_val, (lo, missing, 0))
    else
        return (new_val, (lo, no, n))
    end
end
