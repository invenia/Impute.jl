"""
    NOCB <: Imputor

Fills in missing data using the Next Observation Carried Backward (NOCB) approach.
"""
struct NOCB <: Imputor end

"""
    impute!(imp::NOCB, context::AbstractContext, data::AbstractVector)

Iterates backwards through the `data` and fills missing data with the next
existing observation.

WARNING: missing elements at the tail of the array may not be imputed if there is no
existing observation to carry backward. As a result, this method does not guarantee
that all missing values will be imputed.

# Usage
```

```
"""
function impute!(imp::NOCB, context::AbstractContext, data::AbstractVector)
    context() do c
        end_idx = findlast(c, data) - 1
        for i in end_idx:-1:1
            if ismissing(c, data[i])
                data[i] = data[i+1]
            end
        end

        return data
    end
end
