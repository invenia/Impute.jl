struct Fill{I, V}
    xs::I
    val::V
end

IteratorSize(::Type{<:Fill}) = HasLength()
IteratorEltype(::Type{Fill{I}}) where {I} = IteratorEltype(I)
eltype(::Type{Fill{I}}) where {I} = eltype(I)

function fill(xs::I, value::V) where {I, V}
    T = eltype(I)

    if V <: Union{T, OnlineStat})
        return Fill{I, V}(xs, val)
    else
        throw(ArgumentError("`value` must be of type $T or OnlineStat`"))
    end
end

function iterate(it::Fill, state=nothing)
    # Get the tuple from the underlying iterator
    xs_state = state === nothing ? iterate(it.xs) : iterate(it.xs, tail(state))

    # Exit if nothing
    xs_state === nothing && return nothing

    # Extract the fill value from the state or create a new one.
    it_val = state === nothing ? it.val : first(state)
    xs_item, xs_st = first(xs_state), tail(xs_state)

    # Update the item and fit online stats as necessary
    new_item, new_val = update(it, it_val, xs_item)

    return (new_item, (new_val, xs_state...))
end

# We're just passing fill for dispatch purposes so we can keep using the term update
# for each iterator without naming collisions
function update(it::Fill, val, item)
    new_item = map(item) do x
        ismissing(x) ? val : x
    end

    return (new_item, val)
end

function update(it::Fill, val::OnlineStat, item)
    o = copy(val)

    new_item = map(item) do x
        if ismissing(x)
            value(o)
        else
            fit!(o, x)
            x
        end
    end

    return (new_item, o)
end

function update(it::Fill, val::Group, item)
    g = copy(val)

    new_item = map(zip(g.stats, item)) do (o, x)
        if ismissing(x)
            value(o)
        else
            fit!(o, x)
            x
        end
    end

    return (new_item, g)
end
