"""
    Imputor

An imputor stores information about imputing values in `AbstractArray`s and `Tables.table`s.
New imputation methods are expected to subtype `Imputor` and, at minimum,
implement the `_impute!(data::AbstractArrays, imp::<MyImputor>)` method.

While fallback `impute` and `impute!` methods are provided to extend your `_impute!` methods to
n-dimensional arrays and tables, you can always override these methods to change the
behaviour as necessary.
"""
abstract type Imputor end

#=
These default methods are required because @auto_hash_equals doesn't
play nice with Base.@kwdef
=#
function Base.hash(imp::T, h::UInt) where T <: Imputor
    h = hash(Symbol(T), h)

    for f in fieldnames(T)
        h = hash(getfield(imp, f), h)
    end

    return h
end

function Base.:(==)(a::T, b::T) where T <: Imputor
    result = true

    for f in fieldnames(T)
        if !isequal(getfield(a, f), getfield(b, f))
            result = false
            break
        end
    end

    return result
end

impute_docstring = """
    impute(data::T, imp; dims=:, kwargs...) -> T
    impute!(data::A, imp; dims=:, kwargs...) -> A

Returns a new copy of the `data` with the missing data imputed by the imputor `imp`. If the mutating version
`impute!` is used, it will also update the missing values in-place.

By default, `data` is assumed to be laid out like a `DataFrame`, with each column representing a variable and
each row representing one observation. Other layouts can be handled via the `dims` keyword argument.

# Arguments
* `data`: the data to be impute
* `imp::Imputor`: the Imputor method to use

# Keyword arguments
* `dims = :`: The dimensions to impute along, either `:cols` or `:rows`. If data are in `DataFrame` format,
with variables in columns and observations in rows, use `dims = :cols`. If it is transposed, with variables
in rows and observations in columns, use `dims=:rows`.

# Returns
* `AbstractArray{Union{T, Missing}}`: the input `data` with values imputed. (Mutation isn't guaranteed for 
all array types, so we always return the result).

# NOTES
1. Matrices have a deprecated `dims=2` special case as `dims=:` is a breaking change
2. `eachslice` is used internally which requires Julia 1.1

# Examples
```jldoctest
julia> using Impute: Interpolate, Substitute, impute, impute!

julia> using Statistics: mean

# Linear interpolation in a vector
julia> v = [1.0, 2.0, missing, missing, 5.0]
5-element Vector{Union{Missing, Float64}}:
 1.0
 2.0
  missing
  missing
 5.0

julia> impute(v, Interpolate())
5-element Vector{Union{Missing, Float64}}:
 1.0
 2.0
 3.0
 4.0
 5.0
`
# Usage of the `dims` keyword argument

julia> x = [1.0 missing; missing 2.0]
2×2 Matrix{Union{Missing, Float64}}:
 1.0        missing
  missing  2.0

julia> imp = Substitute(statistic = mean)
Substitute{typeof(mean)}(Statistics.mean)

# Variables in columns, each row is an observation (like a DataFrame)
julia> impute(x, imp, dims=:cols)
2×2 Matrix{Union{Missing, Float64}}:
 1.0  2.0
 1.0  2.0

# Variables in rows, each column is an observation
julia> impute(x, imp, dims=:rows)
2×2 Matrix{Union{Missing, Float64}}:
 1.0  1.0
 2.0  2.0

# Impute over all dimensions
julia> impute(x, imp, dims=:)
2×2 Matrix{Union{Missing, Float64}}:
 1.0  1.5
 1.5  2.0

# Default is `dims = :cols`
julia> impute(x, imp)
2×2 Matrix{Union{Missing, Float64}}:
 1.0  2.0
 1.0  2.0

# In-place imputation

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute!(M, Interpolate(), dims=:rows)
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  5.5

julia> M
2×5 Matrix{Union{Missing, Float64}}:
 1.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  5.5
"""


"""
$impute_docstring
"""
function impute(data, imp::Imputor; kwargs...)
    # NOTE: We don't use a return type declaration here because `trycopy` isn't guaranteed
    # to return the same type passed in. For example, subarrays and subdataframes will
    # return a regular array or dataframe.
    return impute!(trycopy(data), imp; kwargs...)
end

"""
$impute_docstring
"""
function impute!(
    data::A, imp::Imputor; dims=:, kwargs...
)::A where A <: AbstractArray{Union{T, Missing}} where T
    dims === Colon() && return _impute!(data, imp; kwargs...)

    for x in eachslice(data; dims=dims)
        _impute!(x, imp; kwargs...)
    end

    return data
end


function impute!(
    data::M, imp::Imputor; dims=nothing, kwargs...
)::M where M <: AbstractMatrix{Union{T, Missing}} where T
    dims === Colon() && return _impute!(data, imp; kwargs...)
    # We're calling our `dim` function to throw a depwarn if `dims === nothing`
    d = dim(data, dims)

    for x in eachslice(data; dims=d)
        _impute!(x, imp; kwargs...)
    end

    return data
end

impute!(data::AbstractMatrix{Missing}, imp::Imputor; kwargs...) = data

"""
    impute!(data::T, imp; kwargs...) -> T where T <: AbstractVector{<:NamedTuple}

Special case rowtables which are arrays, but we want to fallback to the tables method.
"""
function impute!(data::T, imp::Imputor)::T where T <: AbstractVector{<:NamedTuple}
    return materializer(data)(impute!(Tables.columns(data), imp))
end

"""
    impute!(data::AbstractArray, imp) -> data


Just returns the `data` when the array doesn't contain `missing`s
"""
impute!(data::AbstractArray, imp::Imputor; kwargs...) = disallowmissing(data)

"""
    impute!(data::AbstractArray{Missing}, imp) -> data

Just return the `data` when the array only contains `missing`s
"""
impute!(data::AbstractArray{Missing}, imp::Imputor; kwargs...) = data


"""
    impute!(table, imp; cols=nothing) -> table

Imputes the data in a table by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `table`: the data to impute

# Keyword Arguments
* `cols`: The columns to impute along (default is to impute all columns)

# Returns
* the input `data` with values imputed

# Example
```jldoctest
julia> using DataFrames; using Impute: Interpolate, impute


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

julia> impute(df, Interpolate())
5×2 DataFrame
 Row │ a         b
     │ Float64?  Float64?
─────┼────────────────────
   1 │      1.0       1.1
   2 │      2.0       2.2
   3 │      3.0       3.3
   4 │      4.0       4.4
   5 │      5.0       5.5
```
"""
function impute!(table::T, imp::Imputor; cols=nothing)::T where T
    # TODO: We could probably handle iterators of tables here
    istable(table) || throw(MethodError(impute!, (table, imp)))

    # Extract a columns iterator that we should be able to use to mutate the data.
    # NOTE: Mutation is not guaranteed for all table types, but it avoid copying the data
    columntable = Tables.columns(table)

    cnames = cols === nothing ? propertynames(columntable) : cols
    for cname in cnames
        impute!(getproperty(columntable, cname), imp)
    end

    return table
end

files = [
    "interp.jl",
    "knn.jl",
    "locf.jl",
    "nocb.jl",
    "replace.jl",
    "srs.jl",
    "substitute.jl",
    "svd.jl",
]

for file in files
    include(joinpath("imputors", file))
end
