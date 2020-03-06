"""
    Gaps{I}

An internal iterator that keeps track of missing data gaps.
For each variable we track the last observation (lo), current observation (co)
and gap size.

This is used for defining the LOCF, NOCB and Interpolate iterators.

WARNING: This doesn't really work in the multivariate case because you'd need to have the
interpolation imputor run the gaps imputor over the dataset. This would create unnecessary
copies of the data and would be even worse if you needed to use the `Unzip` iterator to
iterate at different rates for multiple variables.
"""
struct Gaps{I}
    xs::I
end

# Should handle scalar and not scalar iteration for performance reasons.

function ifindnext(it, state=(0,))
    n, xs_state = first(state), tail(state)
    t = iterate(it, xs_state...)
    t === nothing && return nothing
    val, st = Iterators.peel(t)
    return ismissing(val) ? ifindnext(it, (n + 1, st...)) : (val, n)
end

# Is not guaranteed to work on all iterators. Use caution.
ifindprev(it, state) = ifindnext(Iterators.reverse(it), (0, state))
function ifindprev(it::AbstractArray, state)
    ifindnext(
        Iterators.reverse(it),
        (0, reverse(eachindex(it)), state),
    )
end
