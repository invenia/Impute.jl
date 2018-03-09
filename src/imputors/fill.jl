"""
    Fill <: Imputor

Fills in the missing data with a specific value.

# Fields
* `value::Any`: A scalar missing value or a function that returns the a scalar if
    passed the data with missing data removed (e.g, `mean`)
"""
type Fill <: Imputor
    value::Any
end

"""
    Fill() -> Fill

By default `Fill()` will use the mean of the existing values as the fill value.
"""
Fill() = Fill(mean)

"""
    impute!{T<:Any}(imp::Fill, ctx::Context, data::AbstractArray{T, 1})

Computes the fill value if `imp.value` is a `Function` (i.e., `imp.value(drop(copy(data)))`)
and replaces all missing values in the `data` with that value.
"""
function impute!{T<:Any}(imp::Fill, ctx::Context, data::AbstractArray{T, 1})
    fill_val = if isa(imp.value, Function)
        imp.value(drop(copy(data)))
    else
        imp.value
    end

    for i in 1:length(data)
        if ismissing(ctx, data[i])
            data[i] = fill_val
        end
    end

    return data
end
