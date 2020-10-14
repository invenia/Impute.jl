# Generate a functional interface from the Assertion and Imputor types.
"""
    _splitkwargs(::Type{T}, kwargs...) where T -> (imp, rem)

Takes a type with kwargs and returns the constructed type and the
unused kwargs which should be passed to the `impute!` call.

NOTE: This is only intended to be used internally
"""
function _splitkwargs(::Type{T}, kwargs...) where T
    rem = Dict(kwargs...)
    kwdef = empty(rem)

    for f in fieldnames(T)
        if haskey(rem, f)
            kwdef[f] = pop!(rem, f)
        end
    end

    return (T(; kwdef...), rem)
end

# Specialcase kwargs constructor for substitute.
# TODO: Add an imputor method that types should overwrite when necessary or have it fallback to `fieldnames`
function _splitkwargs(::Type{Substitute}, kwargs...)
    rem = Dict(kwargs...)
    kwdef = empty(rem)

    for f in (:statistic, :robust, :weights)
        if haskey(rem, f)
            kwdef[f] = pop!(rem, f)
        end
    end

    return (Substitute(; kwdef...), rem)
end

const global assertion_methods = (
    threshold = Threshold,
)

const global imputation_methods = (
    dropobs = DropObs,
    dropvars = DropVars,
    interp = Interpolate,
    interpolate = Interpolate,
    fill = Fill,
    locf = LOCF,
    nocb = NOCB,
    replace = Replace,
    srs = SRS,
    standardize = Standardize,
    substitute = Substitute,
    svd = SVD,
    knn = KNN,
)

for (func, type) in pairs(assertion_methods)
    typename = nameof(type)
    @eval begin
        function $func(data; kwargs...)
            a, rem = _splitkwargs($typename, kwargs...)
            return assert(data, a; rem...)
        end
    end
end

for (func, type) in pairs(imputation_methods)
    typename = nameof(type)
    func! = Symbol(func, :!)

    @eval begin
        function $func(data; kwargs...)
            imp, rem = _splitkwargs($typename, kwargs...)
            return impute(data, imp; rem...)
        end
        function $func!(data; kwargs...)
            imp, rem = _splitkwargs($typename, kwargs...)
            return impute!(data, imp; rem...)
        end
        @deprecate $func(; kwargs...) data -> $func(data; kwargs...) false
        @deprecate $func!(; kwargs...) data -> $func!(data; kwargs...) false
    end
end

# Provide a specific functional API for Impute.Filter.
filter(data; kwargs...) = apply(data, Filter(); kwargs...)
filter!(data; kwargs...) = apply!(data, Filter(); kwargs...)
filter(f::Function, data; kwargs...) = apply(data, Filter(f); kwargs...)
filter!(f::Function, data; kwargs...) = apply!(data, Filter(f); kwargs...)

@doc """
    Impute.threshold(data; ratio=0.1, weights=nothing, kwargs...)

Assert that proportion of missing values in the `data` do not exceed the `ratio`.

# Examples
```julia-repl
julia> using DataFrames, Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64  │ Float64  │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.threshold(df)
ERROR: ThresholdError: Ratio of missing values exceeded 0.1 (0.4)
Stacktrace:
...

julia> Impute.threshold(df; ratio=0.8)
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64  │ Float64  │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │
```
"""
threshold

@doc """
    Impute.dropobs(data; dims=1)

Removes missing observations from the `AbstractArray` or `Tables.table` provided.
See [DropObs](@ref) for details.

!!! Use `Impute.filter(data; dims=1)` instead.

# Example
```julia-repl
julia> using DataFrames; using Impute: Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64  │ Float64  │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.dropobs(df; dims=2)
3×2 DataFrames.DataFrame
│ Row │ a       │ b       │
│     │ Float64 │ Float64 │
├─────┼─────────┼─────────┤
│ 1   │ 1.0     │ 1.1     │
│ 2   │ 2.0     │ 2.2     │
│ 3   │ 5.0     │ 5.5     │
```
""" dropobs

@doc """
    Impute.dropvars(data; dims=1)

Finds variables with missing values in a `AbstractMatrix` or `Tables.table` and
removes them from the input data. See [DropVars](@ref) for details.

!!! Use `Impute.filter(data; dims=2)` instead.

# Example
```julia-repl
julia> using DataFrames; using Impute: Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.dropvars(df)
0×0 DataFrame
```
""" dropvars

@doc """
    Impute.filter([f,] data; dims)

Filters values, rows, columns or slices of data that should be removed.
The default function `f` will removing `missing`s, or any rows, columns or slices
containing `missing`s.

# Examples
```jldoctest
julia> using DataFrames; using Impute: Impute


julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.filter(df; dims=:cols)
0×0 DataFrame

julia> Impute.filter(df; dims=:rows)
3×2 DataFrame
│ Row │ a       │ b       │
│     │ Float64 │ Float64 │
├─────┼─────────┼─────────┤
│ 1   │ 1.0     │ 1.1     │
│ 2   │ 2.0     │ 2.2     │
│ 3   │ 5.0     │ 5.5     │
```
""" filter

@doc """
    Impute.interp(data; dims=1)

Performs linear interpolation between the nearest values in an vector.
See [Interpolate](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.interp(df)
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 3.0      │ 3.3      │
│ 4   │ 4.0      │ 4.4      │
│ 5   │ 5.0      │ 5.5      │
```
""" interp

@doc """
    Impute.fill(data; value=mean, dims=1)

Fills in the missing data with a specific value. See [Fill](@ref) for details.

!!! Use `Impute.replace` for constants or `Impute.substitue` for median/mode substitution.

# Example
```julia-repl
julia> using DataFrames; using Impute: Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.fill(df; value=-1.0)
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ -1.0     │ 3.3      │
│ 4   │ -1.0     │ -1.0     │
│ 5   │ 5.0      │ 5.5      │
```
""" fill

@doc """
    Impute.locf(data; dims=1)

Iterates forwards through the `data` and fills missing data with the last existing
observation. See [LOCF](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.locf(df)
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 2.0      │ 3.3      │
│ 4   │ 2.0      │ 3.3      │
│ 5   │ 5.0      │ 5.5      │
```
""" locf

@doc """
    Impute.nocb(data; dims=1)

Iterates backwards through the `data` and fills missing data with the next existing
observation. See [LOCF](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.nocb(df)
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 5.0      │ 3.3      │
│ 4   │ 5.0      │ 5.5      │
│ 5   │ 5.0      │ 5.5      │
```
""" nocb

@doc """
    Impute.srs(data; rng=Random.GLOBAL_RNG)

Simple Random Sampling (SRS) imputation is a method for imputing both continuous and
categorical variables. Furthermore, it completes imputation while preserving the
distributional properties of the variables (e.g., mean, standard deviation).

# Example
```julia-repl
julia> using DataFrames; using Random; using Impute: Impute

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.srs(df; rng=MersenneTwister(1234))
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64? │ Float64? │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 1.0      │ 3.3      │
│ 4   │ 2.0      │ 3.3      │
│ 5   │ 5.0      │ 5.5      │
```
""" srs

@doc """
    Impute.standardize(data; values)

Standardize (or replace) various missing data representations with `missing`.

# Keyword Arguments
* `value::Tuple`: A tuple of values that should be considered `missing`

# Example
```jldoctest
julia> using DataFrames, Impute

julia> df = DataFrame(
           :a => [1.1, 2.2, NaN, NaN, 5.5],
           :b => [1, 2, 3, -9999, 5],
           :c => ["v", "w", "x", "y", "NULL"],
       )
5×3 DataFrame
│ Row │ a       │ b     │ c      │
│     │ Float64 │ Int64 │ String │
├─────┼─────────┼───────┼────────┤
│ 1   │ 1.1     │ 1     │ v      │
│ 2   │ 2.2     │ 2     │ w      │
│ 3   │ NaN     │ 3     │ x      │
│ 4   │ NaN     │ -9999 │ y      │
│ 5   │ 5.5     │ 5     │ NULL   │

julia> Impute.standardize(df; values=(NaN, -9999, "NULL"))
5×3 DataFrame
│ Row │ a        │ b       │ c       │
│     │ Float64? │ Int64?  │ String? │
├─────┼──────────┼─────────┼─────────┤
│ 1   │ 1.1      │ 1       │ v       │
│ 2   │ 2.2      │ 2       │ w       │
│ 3   │ missing  │ 3       │ x       │
│ 4   │ missing  │ missing │ y       │
│ 5   │ 5.5      │ 5       │ missing │
```
""" standardize

@doc """
    Impute.substitute(data; statistic=nothing)
    Impute.substitute(data; robust=true, weights=nothing)

Substitute missing values with a summary statistic over the non-missing values.

# Keyword Arguments
* `statistic`: A summary statistic function to be applied to the non-missing values.
  This function should return a value of the same type as the input data `eltype`.
  If this function isn't passed in then the `defaultstats` function is used to make a
  best guess.
* `robust`: Whether to use `median` or `mean` for continuous datasets in `defaultstats`
* `weights`: A set of statistical weights to apply to the `mean` or `median` in `defaultstats`.

See [Substitute](@ref) for details on substitution rules defined in `defaultstats`.

# Example
```jldoctest
julia> using DataFrames, Impute

julia> df = DataFrame(
                  :a => [8.9, 2.2, missing, missing, 1.3, 6.2, 3.7, 4.8],
                  :b => [2, 6, 3, missing, 7, 1, 9, missing],
                  :c => [true, false, true, true, false, missing, false, true],
              )
8×3 DataFrame
│ Row │ a        │ b       │ c       │
│     │ Float64? │ Int64?  │ Bool?   │
├─────┼──────────┼─────────┼─────────┤
│ 1   │ 8.9      │ 2       │ 1       │
│ 2   │ 2.2      │ 6       │ 0       │
│ 3   │ missing  │ 3       │ 1       │
│ 4   │ missing  │ missing │ 1       │
│ 5   │ 1.3      │ 7       │ 0       │
│ 6   │ 6.2      │ 1       │ missing │
│ 7   │ 3.7      │ 9       │ 0       │
│ 8   │ 4.8      │ missing │ 1       │

julia> Impute.substitute(df)
8×3 DataFrame
│ Row │ a        │ b      │ c     │
│     │ Float64? │ Int64? │ Bool? │
├─────┼──────────┼────────┼───────┤
│ 1   │ 8.9      │ 2      │ 1     │
│ 2   │ 2.2      │ 6      │ 0     │
│ 3   │ 4.25     │ 3      │ 1     │
│ 4   │ 4.25     │ 4      │ 1     │
│ 5   │ 1.3      │ 7      │ 0     │
│ 6   │ 6.2      │ 1      │ 1     │
│ 7   │ 3.7      │ 9      │ 0     │
│ 8   │ 4.8      │ 4      │ 1     │
```
""" substitute

@doc """
    Impute.knn(; k=1, threshold=0.5, dist=Euclidean())

Imputation using k-Nearest Neighbor algorithm.

# Keyword Arguments
* `k::Int`: number of nearest neighbors
* `dist::MinkowskiMetric`: distance metric suppports by `NearestNeighbors.jl` (Euclidean, Chebyshev, Minkowski and Cityblock)
* `threshold::AbsstractFloat`: thershold for missing neighbors

# Reference
* Troyanskaya, Olga, et al. "Missing value estimation methods for DNA microarrays." Bioinformatics 17.6 (2001): 520-525.

# Example
```jldoctest
julia> using Impute, Missings

julia> data = allowmissing(reshape(sin.(1:20), 5, 4)); data[[2, 3, 7, 9, 13, 19]] .= missing; data
5×4 Array{Union{Missing, Float64},2}:
  0.841471  -0.279415  -0.99999   -0.287903
   missing    missing  -0.536573  -0.961397
   missing   0.989358    missing  -0.750987
 -0.756802    missing   0.990607    missing
 -0.958924  -0.544021   0.650288   0.912945

julia> result = Impute.knn(data; dims=:cols)
5×4 Array{Union{Missing, Float64},2}:
  0.841471  -0.279415  -0.99999    -0.287903
 -0.756802  -0.279415  -0.536573   -0.961397
 -0.756802   0.989358   0.0568575  -0.750987
 -0.756802  -0.279415   0.990607   -0.519445
 -0.958924  -0.544021   0.650288    0.912945
```
""" knn

@doc """
    Impute.svd(; kwargs...)

Imputes the missing values in a matrix using an expectation maximization (EM) algorithm
over low-rank SVD approximations.

# Keyword Arguments
* `init::Imputor`: initialization method for missing values (default: Substitute())
* `rank::Union{Int, Nothing}`: rank of the SVD approximation (default: nothing meaning start and 0 and increase)
* `tol::Float64`: convergence tolerance (default: 1e-10)
* `maxiter::Int`: Maximum number of iterations if convergence is not achieved (default: 100)
* `limits::Unoin{Tuple{Float64, Float64}, Nothing}`: Bound the possible approximation values (default: nothing)
* `verbose::Bool`: Whether to display convergence progress (default: true)

# References
* Troyanskaya, Olga, et al. "Missing value estimation methods for DNA microarrays." Bioinformatics 17.6 (2001): 520-525.

# Example
```jldoctest
julia> using Impute, Missings

julia> data = allowmissing(reshape(sin.(1:20), 5, 4)); data[[2, 3, 7, 9, 13, 19]] .= missing; data
5×4 Array{Union{Missing, Float64},2}:
  0.841471  -0.279415  -0.99999   -0.287903
   missing    missing  -0.536573  -0.961397
   missing   0.989358    missing  -0.750987
 -0.756802    missing   0.990607    missing
 -0.958924  -0.544021   0.650288   0.912945

julia> result = Impute.svd(data; dims=:cols)
5×4 Array{Union{Missing, Float64},2}:
  0.841471  -0.279415  -0.99999   -0.287903
  0.220258   0.555829  -0.536573  -0.961397
 -0.372745   0.989358   0.533193  -0.750987
 -0.756802   0.253309   0.990607   0.32315
 -0.958924  -0.544021   0.650288   0.912945
```
""" svd
