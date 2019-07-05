"""
    Fill <: Imputor

Fills in the missing data with a specific value.

# Fields
* `value::Any`: A scalar missing value or a function that returns the a scalar if
  passed the data with missing data removed (e.g, `mean`)
* `context::AbstractContext`: A context which keeps track of missing data
  summary information
"""
struct Fill{T} <: Imputor
    value::T
    context::AbstractContext
end

"""Fill(; value=mean, context=Context()) -> Fill"""
Fill(; value=mean, context=Context()) = Fill(value, context)

"""
    impute!(imp::Fill, data::AbstractVector)

Computes the fill value if `imp.value` is a `Function` (i.e., `imp.value(drop(copy(data)))`)
and replaces all missing values in the `data` with that value.
"""
function impute!(imp::Fill, data::AbstractVector)
    imp.context() do c
        fill_val = if isa(imp.value, Function)
            imp.value(Iterators.drop(copy(data); context=c))
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
