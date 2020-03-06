module Iter

using OnlineStatsBase

import Base: iterate, eltype, length, size, tail
import Base: IteratorSize, IteratorEltype
import Base: SizeUnknown, IsInfinite, HasLength, HasShape
import Base: HasEltype, EltypeUnknown

# Some common types that we'll consider scalar
isscalar(x::Missing) = true
isscalar(x::Nothing) = true
isscalar(x::Number) = true
isscalar(x::Char) = true
isscalar(x::Symbol) = true

# This decision is specific to the context of imputation.
isscalar(x::AbstractString) = true

# Some types that we know aren't scalar
isscalar(x::Tuple) = false
isscalar(x::AbstractArray) = false
isscalar(x::NamedTuple) = false

# Somewhat slow fallback which tests if we can iterate over the type.
isscalar(x::T) where {T} = !method_exists(iterate, (T,))

# If the type is Any then we really can't say it's scalar.
isscalar(x::Any) = false


for file in ("threshold.jl", "drop.jl", "fill.jl", "interp.jl", "locf.jl", "nocb.jl")
    include(joinpath("iter", file))
end

ITER_TYPES = Union{Threshold, Iterpolate, LOCF, NOCB}

# Generic iterate method which can handle scalar and non-scalar iteration.
# Requirements:
# - Implemented `_update(it, val::Scalar, it_st, xs_st)`
# - state is of the form `(it_st, xs_st)`
# - internal iterator is a field called `xs`
function iterate(it::iter_types)
    t = @ifsomething iterate(it.xs)
    val, xs_st = Iterator.peel(t)

    if isscalar(val)
        return _update(it, val, xs_st)
    else
        results = map(val) do x
            _update(it, x, xs_st)
        end

        new_val, st = unzip(results)
        # We're assuming that the original iterable type can be reconstructed from a vector
        # of the values:
        # - Array{T, 1}([...])
        # - Tuple{A, B, C}([...])
        # - NamedTuple{(:a, :b, :c), Tuple{A, B, C}}([...])
        return typeof(val)(new_val), st
    end
end

function iterate(it::ITER_TYPES, state)
    it_st, xs_st = Iterators.peel(state)
    t = @ifsomething iterate(it.xs, xs_st)
    val, xs_st = Iterators.peel(t)

    if isscalar(val)
        return _update(it, val, it_st, xs_st)
    else
        results = map(zip(val, it_st)) do (x, s)
            # TODO: How do we handle search forward for interplolation?
            _update(it, x, s, xs_st)
        end

        new_val, st = unzip(results)
        # We're assuming that the original iterable type can be reconstructed from a vector
        # of the values:
        # - Array{T, 1}([...])
        # - Tuple{A, B, C}([...])
        # - NamedTuple{(:a, :b, :c), Tuple{A, B, C}}([...])
        return typeof(val)(new_val), st
    end
end
