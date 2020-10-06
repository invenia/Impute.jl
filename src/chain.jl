const Transform = Union{Assertion, Filter, Imputor}

"""
    Chain

Runs multiple `Assertions`, `Filter` or `Imputor`s on the same data in the order they're
provided.

# Fields
* `transforms::Vector{Union{Assertion, Filter, Imputor}}`
"""
struct Chain
    transforms::Vector{Transform}
end

"""
    Chain(transforms::Union{Assertion, Filter, Imputor}...) -> Chain

Creates a Chain using the transforms provided (ordering matters).
"""
Chain(transforms::Transform...) = Chain(collect(transforms))

"""
Compose new chains with the composition operator

# Example

```jldoctest
julia> using Impute: Impute, Interpolate, NOCB, LOCF

julia> M = [missing 2.0 missing missing 5.0; 1.1 2.2 missing 4.4 missing]
2×5 Array{Union{Missing, Float64},2}:
  missing  2.0  missing   missing  5.0
 1.1       2.2  missing  4.4        missing

julia> Impute.run(M, Interpolate() ∘ NOCB() ∘ LOCF(); dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 2.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  4.4
```
"""
Base.:(∘)(a::Transform, b::Transform) = Chain(Transform[a, b])
function Base.:(∘)(C::Chain, b::Transform)
    push!(C.transforms, b)
    return C
end

"""
    run(data, C::Chain; kwargs...)

Runs the transforms over the supplied data.

# Arguments
* `data`: our data to impute
* `C::Chain`: the chain to run


# Keyword Arguments
* `kwargs`: Keyword arguments that should be applied to each transform (ex `dims=:cols`)

# Returns
* our imputed data
"""
function run(data, C::Chain; kwargs...)
    # Since some operation like filtering can't consistently mutate the data we make a copy
    # and don't support a mutating form.
    X = trycopy(data)

    for t in C.transforms
        if isa(t, Assertion)
            # Assertions just return the input
            assert(X, t; kwargs...)
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
