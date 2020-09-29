"""
    Assertion

An Assertion stores settings for checking the validity of a `AbstractArray` or `Tables.table` containing missing values.
New assertions are expected to subtype `Impute.Assertion` and, at minimum,
implement the `assert(data::AbstractVector{Union{T, Missing}}, ::<MyAssertion>)` method.
"""
abstract type Assertion end


"""
    assert(data, a::Assertion; kwargs...)

If the assertion `a` fails then an `AssertionError` is thrown, otherwise the `data`
provided is returned without mutation.

# Keywords
* `dims`: The dimension to apply the assertion along (e.g., observations dim)
"""
assert

# A couple fallbacks for matrices and tables
function assert(data::AbstractMatrix, a::Assertion; dims=1)
    for var in varwise(data; dims=dims)
        assert(var, a)
    end
    return data
end

function assert(table, a::Assertion)
    istable(table) || throw(MethodError(a, (table, a)))
    columntable = Tables.columns(table)

    for cname in propertynames(columntable)
        assert(getproperty(columntable, cname), a)
    end

    return table
end

include("assertions/threshold.jl")
