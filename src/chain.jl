const Transform = Union{Validator, Filter, Imputor}

"""
    Chain{T<:Tuple{Vararg{Transform}}} <: Function

Runs multiple `Validators`, `Filter` or `Imputor`s on the same data in the order they're
provided.

# Fields
* `transforms::Vector{Union{Validator, Filter, Imputor}}`
"""
struct Chain{T<:Tuple{Vararg{Transform}}} <: Function
    transforms::T
end

Chain(transforms::Vector{<:Transform}) = Chain(Tuple(transforms))

"""
    Chain(transforms::Union{Validator, Filter, Imputor}...) -> Chain

Creates a Chain using the transforms provided (ordering matters).
"""
Chain(transforms::Transform...) = Chain(tuple(transforms...))

"""
Compose new chains with the composition operator

# Example

```jldoctest
julia> using Impute: Impute, Interpolate, NOCB, LOCF

julia> M = [missing 2.0 missing missing 5.0; 1.1 2.2 missing 4.4 missing]
2×5 Array{Union{Missing, Float64},2}:
  missing  2.0  missing   missing  5.0
 1.1       2.2  missing  4.4        missing

julia> C = Interpolate() ∘ NOCB() ∘ LOCF();

julia> C(M; dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 2.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  4.4
```
"""
Base.:(∘)(a::Transform, b::Transform) = Chain(a, b)
Base.:(∘)(C::Chain, b::Transform) = Chain(C.transforms..., b)

"""
    (C::Chain)(data; kwargs...)

Runnable the "callable" chain `C` on the supplied `data`.

# Arguments
* `data`: our data to impute

# Keyword Arguments
* `kwargs`: Keyword arguments that should be applied to each transform (ex `dims=:cols`)

# Returns
* our imputed data
"""
function (C::Chain)(data; kwargs...)
    # Since some operation like filtering can't consistently mutate the data we make a copy
    # and don't support a mutating form.
    X = trycopy(data)

    for t in C.transforms
        if isa(t, Validator)
            # Validators just return the input
            validate(X, t; kwargs...)
        elseif isa(t, Filter)
            # Filtering doesn't always work in-place
            X = apply(X, t; kwargs...)
        else
            # An in-place impute! method should always exist
            X = impute!(X, t; kwargs...)
        end
    end

    return X
end
