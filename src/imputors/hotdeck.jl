@auto_hash_equals struct HotDeck <: Imputor
   vardim::Int
   context::AbstractContext
end


"""
    HotDeck(; vardim = 2, context = Context())

Hot deck imputation is a method for imputing both continuous and categorical
variables. Furthermore, it completes imputation while preserving the distributional 
properties of the variables (e.g., mean, standard deviation). 

The basic idea is that for a given variable, `x`, with missing data, we randomly draw 
from the observed values of `x` to impute the missing elements. Since the random draws
from `x` for imputation are done in proportion to the frequency distribution of the values 
in `x`, the univariate distributional properties are generally not impacted; this is true 
for both categorical and continuous data.


# Keyword Arguments
* `vardim = 2::Int`: Specify the dimension for variables in matrix input data
* `context::AbstractContext`: A context which keeps track of missing data
  summary information

# Example
```jldoctest
julia> using Impute: HotDeck, Context, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, HotDeck(; vardim = 1, context = Context(; limit = 1.0)))
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  5.0  2.0  5.0
 1.1  2.2  3.3  3.3  5.5
```
"""
HotDeck(; vardim = 2, context = Context()) = HotDeck(vardim, context)


function impute!(data::AbstractVector, imp::HotDeck)
    obs_values = collect(skipmissing(data))
    imp.context() do c
        for i = 1:lastindex(data)
            if ismissing(c, data[i])
                data[i] = rand(obs_values)
            end
        end

        return data
    end
end
