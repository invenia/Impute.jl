"""
    Assertion

An Assertion stores settings for checking the validity of a `AbstractArray` or `Tables.table` containing missing values.
New assertions are expected to subtype `Impute.Assertion` and, at minimum,
implement the `_assert(data::AbstractArray{Union{T, Missing}}, ::<MyAssertion>)` method.
"""
abstract type Assertion end

"""
    assert(data::AbstractArray, a::Assertion; dims=:)

If the assertion `a` fails then an `ThresholdError` is thrown, otherwise the `data`
provided is returned without mutation. See [`Assertion`](@ref) for the minimum internal
`_assert` call requirements.

# Arguments
* `data::AbstractArray`: the data to be impute along dimensions `dims`
* `a::Assertion`: the assertion to apply

# Keywords
* `dims`: The dimension to apply the `_assert` along (default is `:`)

# Returns
* the input `data` if no error is thrown.

# Throws
* An error when the test fails

```jldoctest
julia> using Test; using Impute: Threshold, ThresholdError, assert

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> @test_throws ThresholdError assert(M, Threshold())
Test Passed
      Thrown: ThresholdError
```
"""
function assert(data::AbstractArray, a::Assertion; dims=:, kwargs...)
    dims === Colon() && return _assert(data, a; kwargs...)
    d = Impute.dim(data, dims)

    for d in eachslice(data; dims=d)
        _assert(d, a; kwargs...)
    end
    return data
end

"""
    assert(table, a::Assertion; cols=nothing)

Applies the assertion `a` to the `table` 1 column at a time; if this is not the desired
behaviour custom `assert` methods should overload this method. See [`Assertion`](@ref) for
the minimum internal `_assert` call requirements.

# Arguments
* `table`: the data to impute
* `a`: the assertion to apply

# Keyword Arguments
* `cols`: The columns to impute along (default is to impute all columns)

# Returns
* the input `data` if no error is thrown.

# Throws
* An error when any column doesn't pass the test

# Example
```jldoctest
julia> using DataFrames, Test; using Impute: Threshold, ThresholdError, assert

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> @test_throws ThresholdError assert(df, Threshold())
Test Passed
      Thrown: ThresholdError
```
"""
function assert(table, a::Assertion; cols=nothing, kwargs...)
    istable(table) || throw(MethodError(assert, (table, a)))
    columntable = Tables.columns(table)

    cnames = cols === nothing ? propertynames(columntable) : cols
    for cname in cnames
        _assert(getproperty(columntable, cname), a; kwargs...)
    end

    return table
end

include("assertions/threshold.jl")
