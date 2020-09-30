"""
    Imputor

An imputor stores information about imputing values in `AbstractArray`s and `Tables.table`s.
New imputation methods are expected to sutype `Imputor` and, at minimum,
implement the `impute!(imp::<MyImputor>, data::AbstractVector)` method.
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

"""
    impute(data::T, imp::Imputor; dims, kwargs...) -> T

Returns a new copy of the `data` with the missing data imputed by the imputor `imp`.
For matrices and tables, data is imputed one variable/column at a time.
If this is not the desired behaviour then you should overload this method or specify a different `dims` value.

# Keywords
* `dims`: The dimension to impute along

# Arguments
* `data: the data to impute
* `imp::Imputor`: the Imputor method to use

# Returns
* the input `data` with values imputed

# Example
```jldoctest
julia> using Impute: Interpolate, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Interpolate(); dims=2)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  5.5
```
"""
function impute(data, imp::Imputor; kwargs...)
    # Not all types support copy, so we fallback to deepcopy
    result = try
        copy(data)
    catch
        deepcopy(data)
    end

    return impute!(result, imp; kwargs...)
end

# Generic fallback for methods that have only defined _impute!(v, imp; kwargs...)
function impute!(data::AbstractArray{Union{T, Missing}}, imp::Imputor; dims=:, kwargs...) where T
    if dims === Colon()
        _impute!(data, imp; kwargs...)
    else
        for x in eachslice(data; dims=dims)
            _impute!(x, imp; kwargs...)
        end
    end
    return data
end

# Used to maintain the backwards compatibility with the previous columnwise default behaviour.
function impute!(data::AbstractMatrix{Union{T, Missing}}, imp::Imputor; dims=nothing, kwargs...) where T
    if dims === nothing
        msg = string(
            "Imputation on a matrix will require specifying `dims=:cols` or `dims=2` in a ",
            "future release",
        )
        Base.depwarn(msg, :impute!)
        dims = :cols
    end

    if dims === :rows || dims == 1
        for x in eachrow(data)
            _impute!(x, imp; kwargs...)
        end
    elseif dims === :cols || dims == 2
        for x in eachcol(data)
            _impute!(x, imp; kwargs...)
        end
    else
        for x in eachslice(data; dims=dims)
            _impute!(x, imp; kwargs...)
        end
    end
    return data
end

# Fallback for rowtables
function impute!(data::AbstractVector{<:NamedTuple}, imp::Imputor)
    return materializer(data)(impute!(Tables.columns(data), imp))
end

# Fallback for arrays of scalars with no missing values
impute!(data::AbstractArray, imp::Imputor; kwargs...) = data


"""
    impute!(table, imp::Imputor)

Imputes the data in a table by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `table`: the data to impute

# Returns
* the input `data` with values imputed

# Example
``jldoctest
julia> using DataFrames; using Impute: Interpolate, impute
julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64  │ Float64  │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> impute(df, Interpolate())
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64  │ Float64  │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 3.0      │ 3.3      │
│ 4   │ 4.0      │ 4.4      │
│ 5   │ 5.0      │ 5.5      │
"""
function impute!(table, imp::Imputor; cols=nothing)
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

for file in ("drop.jl", "locf.jl", "nocb.jl", "interp.jl", "fill.jl", "chain.jl", "srs.jl", "svd.jl", "knn.jl")
    include(joinpath("imputors", file))
end
