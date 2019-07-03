"""
    Fill <: Imputor

Fills in the missing data with a specific value.

# Fields
* `value::Any`: A scalar missing value or a function that returns the a scalar if
    passed the data with missing data removed (e.g, `mean`)
"""
struct Fill{T} <: Imputor
    value::T
end

"""
    Fill() -> Fill

By default `Fill()` will use the mean of the existing values as the fill value.
"""
Fill() = Fill(mean)

"""
    impute!(imp::Fill, context::AbstractContext, data::AbstractVector)

Computes the fill value if `imp.value` is a `Function` (i.e., `imp.value(drop(copy(data)))`)
and replaces all missing values in the `data` with that value.
"""
function impute!(imp::Fill, context::AbstractContext, data::AbstractVector)
    context() do c
        fill_val = if isa(imp.value, Function)
            imp.value(Iterators.drop(copy(data)))
        else
            imp.value
        end

        for i in 1:length(data)
            if ismissing(c, data[i])
                data[i] = fill_val
            end
        end

        return data
    end
end
