"""
    Imputor

An imputor stores information about imputing values in `AbstractArray`s and `Tables.table`s.
New imputation methods are expected to sutype `Imputor` and, at minimum,
implement the `impute!{T<:Any}(imp::<MyImputor>, ctx::Context, data::AbstractArray{T, 1})`
method.
"""

abstract type Imputor end


"""
    impute(imp::Imputor, data)

Copies the `data` before calling the corresponding `impute!(imp, ...)` call.
"""
function impute(imp::Imputor, data)
    impute!(imp, deepcopy(data))
end

"""
    impute!(imp::Imputor, data::AbstractMatrix)

Imputes the data in a matrix by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `data::AbstractMatrix`: the data to impute

# Returns
* `AbstractMatrix`: the input `data` with values imputed
"""
function impute!(imp::Imputor, data::AbstractMatrix)
    for i in 1:size(data, 2)
        impute!(imp, view(data, :, i))
    end
    return data
end

"""
    impute!(imp::Imputor, table)

Imputes the data in a table by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `table`: the data to impute

# Returns
* the input `data` with values imputed
"""
function impute!(imp::Imputor, table)
    @assert istable(table)
    # Extract a columns iterate that we should be able to use to mutate the data.
    # NOTE: Mutation is not guaranteed for all table types, but it avoid copying the data
    columntable = Tables.columns(table)

    for cname in propertynames(columntable)
        impute!(imp, getproperty(columntable, cname))
    end

    return table
end


for file in ("drop.jl", "locf.jl", "nocb.jl", "interp.jl", "fill.jl", "chain.jl")
    include(joinpath("imputors", file))
end
