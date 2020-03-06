# A threshold iterator describes logic for identifying:
#
# 1. When a dataset has too many missing values
#   - α between 0..1 where n = length otherwise it's window size
#   - α > 1 then it's simply a static number
#
# 2. Defaults to erroring, but could also warn.
#
# Pros:
# - This is relatively simple and allows greater control over when we're checking.
# - If you want to change the behaviour then just write a different iterator.

struct ThresholdError <: Exception
    limit::Float64
    ratio::Float64
end

function Base.showerror(io::IO, err::ThresholdError)
    println(
        io,
        "ThresholdError: More than $(err.limit * 100)% of values " *
        "were missing ($(err.ratio * 100)%)."
    )
end

struct Threshold{I}
    xs::I
    τ::Float64
end

Base.IteratorSize(::Type{<:Threshold}) = HasLength()
Base.IteratorEltype(::Type{Threshold{I}}) where {I} = IteratorEltype(I)
Base.eltype(::Type{Threshold{I}}) where {I} = eltype(I)

function threshold(xs, τ::Float64)
    # TODO: Add a block based threshold n missing per b observations.
    isa(IteratorSize(xs), Union{HasLength, HasShape}) || throw(ArgumentError(
        "Thresholds currently only operate on iterators with a length."
    ))

    return Threshold(xs, τ * length(xs))
end

function Base.iterate(it::Threshold, state=(0, ))
    n, xs_state = first(state), tail(state)
    next_iter = iterate(it.xs, xs_state...)

    if next_iter === nothing
        return nothing
    else
        n, val, st = update(it, n, first(next_iter), tail(next_iter))

        if n > it.τ
            l = length(it.xs)
            throw(ThresholdError(it.τ / l, n / l))
        end

        return (val, (n, st...))
    end
end

# Special case Missing because `iterate(Missing)` is not defined on 1.3
update(it::Threshold, n::Int, val::Missing, st) = (n + 1, val, st)
update(it::Threshold, n::Int, val, st) = (n + count(ismissing, val), val, st)

# TODO: Support τ being a function
# - Handle rows and columns
# - Support arbitrary thresholding behaviour
