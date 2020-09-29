"""
    Fill(; value=mean)

Fills in the missing data with a specific value.
The current implementation is univariate, so each variable in a table or matrix will
be handled independently.

# Keyword Arguments
* `value::Any`: A scalar or a function that returns a scalar if
  passed the data with missing data removed (e.g, `mean`)

# Example
```jldoctest
julia> using Impute: Fill, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Fill(); dims=2)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.66667  2.66667  5.0
 1.1  2.2  3.3      3.025    5.5
```
"""
struct Fill{T} <: Imputor
    value::T
end

# TODO: Switch to using Base.@kwdef on 1.1
Fill(; value=mean) = Fill(value)

function _impute!(data::AbstractVector, imp::Fill)
    fill_val = if isa(imp.value, Function)
        available = Impute.drop(data)

        if isempty(available)
            @debug "Cannot apply fill function $(imp.value) as all values are missing"
            return data
        else
            imp.value(available)
        end
    else
        imp.value
    end

    for i in eachindex(data)
        if ismissing(data[i])
            data[i] = fill_val
        end
    end

    return data
end
