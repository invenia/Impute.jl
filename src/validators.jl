"""
    Validator

An Validator stores settings for checking the validity of a `AbstractArray` or `Tables.table` containing missing values.
New validations are expected to subtype `Impute.Validator` and, at minimum,
implement the `_validate(data::AbstractArray{Union{T, Missing}}, ::<MyValidator>)` method.
"""
abstract type Validator end

"""
    validate(data::AbstractArray, v::Validator; dims=:)

If the validator `v` fails then an error is thrown, otherwise the `data`
provided is returned without mutation. See [`Validator`](@ref) for the minimum internal
`_validate` call requirements.

# Arguments
* `data::AbstractArray`: the data to be impute along dimensions `dims`
* `v::Validator`: the validator to apply

# Keywords
* `dims`: The dimension to apply the `_validate` along (default is `:`)

# Returns
* the input `data` if no error is thrown.

# Throws
* An error when the test fails

```jldoctest
julia> using Test; using Impute: Threshold, ThresholdError, validate

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> @test_throws ThresholdError validate(M, Threshold())
Test Passed
      Thrown: ThresholdError
```
"""
function validate(data::AbstractArray, a::Validator; dims=:, kwargs...)
    dims === Colon() && return _validate(data, a; kwargs...)
    d = Impute.dim(data, dims)

    for d in eachslice(data; dims=d)
        _validate(d, a; kwargs...)
    end
    return data
end

"""
    validate(table, v::Validator; cols=nothing)

Applies the validator `v` to the `table` 1 column at a time; if this is not the desired
behaviour custom `validate` methods should overload this method. See [`Validator`](@ref) for
the minimum internal `_validate` call requirements.

# Arguments
* `table`: the data to impute
* `v`: the validator to apply

# Keyword Arguments
* `cols`: The columns to impute along (default is to impute all columns)

# Returns
* the input `data` if no error is thrown.

# Throws
* An error when any column doesn't pass the test

# Example
```jldoctest
julia> using DataFrames, Test; using Impute: Threshold, ThresholdError, validate


julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
 Row │ a          b
     │ Float64?   Float64?
─────┼──────────────────────
   1 │       1.0        1.1
   2 │       2.0        2.2
   3 │ missing          3.3
   4 │ missing    missing
   5 │       5.0        5.5

julia> @test_throws ThresholdError validate(df, Threshold())
Test Passed
      Thrown: ThresholdError
```
"""
function validate(table, v::Validator; cols=nothing, kwargs...)
    istable(table) || throw(MethodError(validate, (table, v)))
    columntable = Tables.columns(table)

    cnames = cols === nothing ? propertynames(columntable) : cols
    for cname in cnames
        _validate(getproperty(columntable, cname), v; kwargs...)
    end

    return table
end

include("validators/threshold.jl")
