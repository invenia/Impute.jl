"""
    NOCB <: Imputor

Fills in missing data using the Next Observation Carried Backward (NOCB) approach.
"""
type NOCB <: Imputor end

"""
    impute!{T<:Any}(imp::NOCB, ctx::Context, data::AbstractArray{T, 1})

Iterates backwards through the `data` and fills missing data with the next
existing observation.

WARNING: missing elements at the tail of the array may not be imputed if there is no
existing observation to carry backward. As a result, this method does not guarantee
that all missing values will be imputed.

# Usage
```

```
"""
function impute!{T<:Any}(imp::NOCB, ctx::Context, data::AbstractArray{T, 1})
    end_idx = findlast(ctx, data) - 1
    for i in end_idx:-1:1
        if ismissing(ctx, data[i])
            data[i] = data[i+1]
        end
    end

    return data
end
