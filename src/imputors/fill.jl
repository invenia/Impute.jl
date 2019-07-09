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

Fill in missing values with a values determined by `imp.value`.
If `imp.value`  is a function then the fill values calculated by invoking that function on
the collection of all nonmissing values.
"""
function impute!(imp::Fill, data::AbstractVector)
    imp.context() do c
        fill_val = if isa(imp.value, Function)
            # Call `deepcopy` because we can trust that it's available for all types.
            imp.value(Iterators.drop(deepcopy(data); context=c))
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
