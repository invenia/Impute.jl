type LOCF <: Imputor end

"""
    impute!{T<:Any}(imp::LOCF, ctx::Context, data::AbstractArray{T, 1})

Iterates forwards through the `data` and fills missing data with the last
existing observation.

WARNING: missing elements at the head of the array may not be imputed if there is no
existing observation to carry forward. As a result, this method does not guarantee
that all missing values will be imputed.

# Usage
```

```
"""
function impute!{T<:Any}(imp::LOCF, ctx::Context, data::AbstractArray{T, 1})
    start_idx = findfirst(ctx, data) + 1
    for i in start_idx:length(data)
        if is_missing(ctx, data[i])
            data[i] = data[i-1]
        end
    end

    return data
end
