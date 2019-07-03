struct LOCF <: Imputor end

"""
    impute!(imp::LOCF, context::AbstractContext, data::AbstractVector)

Iterates forwards through the `data` and fills missing data with the last
existing observation.

WARNING: missing elements at the head of the array may not be imputed if there is no
existing observation to carry forward. As a result, this method does not guarantee
that all missing values will be imputed.

# Usage
```

```
"""
function impute!(imp::LOCF, context::AbstractContext, data::AbstractVector)
    context() do c
        start_idx = findfirst(c, data) + 1
        for i in start_idx:length(data)
            if ismissing(c, data[i])
                data[i] = data[i-1]
            end
        end

        return data
    end
end
