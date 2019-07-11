"""
    Imputor

An imputor stores information about imputing values in `AbstractArray`s and `Tables.table`s.
New imputation methods are expected to sutype `Imputor` and, at minimum,
implement the `impute!(imp::<MyImputor>, data::AbstractVector)` method.
"""
abstract type Imputor end

# A couple utility methods to avoid messing up var and obs dimensions
obsdim(imp::Imputor) = imp.vardim == 1 ? 2 : 1
vardim(imp::Imputor) = imp.vardim

function obswise(imp::Imputor, data::AbstractMatrix)
    (imp.vardim == 1 ? view(data, :, i) : view(data, i, :) for i in axes(data, obsdim(imp)))
end

function varwise(imp::Imputor, data::AbstractMatrix)
    (imp.vardim == 1 ? view(data, i, :) : view(data, :, i) for i in axes(data, vardim(imp)))
end

function filterobs(f::Function, imp::Imputor, data::AbstractMatrix)
    mask = [f(x) for x in obswise(imp, data)]
    return imp.vardim == 1 ? data[:, mask] : data[mask, :]
end

function filtervars(f::Function, imp::Imputor, data::AbstractMatrix)
    mask = [f(x) for x in varwise(imp, data)]
    return imp.vardim == 1 ? data[mask, :] : data[:, mask]
end

"""
    impute(data, imp::Imputor)

Returns a new copy of the `data` with the missing data imputed by the imputor `imp`.
"""
function impute(data, imp::Imputor)
    # Call `deepcopy` because we can trust that it's available for all types.
    return impute!(deepcopy(data), imp)
end


# This is a necessary fallback because the tables method doesn't have a type declaration
impute!(data::AbstractVector, imp::Imputor) = MethodError(impute!, (data, imp))

"""
    impute!(data::AbstractMatrix, imp::Imputor)

Imputes the data in a matrix by imputing the values 1 variable at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `data::AbstractMatrix`: the data to impute
* `imp::Imputor`: the Imputor method to use

# Returns
* `AbstractMatrix`: the input `data` with values imputed

# Example
```jldoctest
julia> using Impute: Interpolate, Context, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Interpolate(; vardim=1, context=Context(; limit=1.0)))
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  5.5
```
"""
function impute!(data::AbstractMatrix, imp::Imputor)
    for var in varwise(imp, data)
        impute!(var, imp)
    end
    return data
end

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
julia> using DataFrames; using Impute: Interpolate, Context, impute
julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> impute(df, Interpolate(; vardim=1, context=Context(; limit=1.0)))
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 3.0      │ 3.3      │
│ 4   │ 4.0      │ 4.4      │
│ 5   │ 5.0      │ 5.5      │
"""
function impute!(table, imp::Imputor)
    @assert istable(table)
    # Extract a columns iterate that we should be able to use to mutate the data.
    # NOTE: Mutation is not guaranteed for all table types, but it avoid copying the data
    columntable = Tables.columns(table)

    for cname in propertynames(columntable)
        impute!(getproperty(columntable, cname), imp)
    end

    return table
end


for file in ("drop.jl", "locf.jl", "nocb.jl", "interp.jl", "fill.jl", "chain.jl")
    include(joinpath("imputors", file))
end
