# Generate a functional interface from the Assertion and Imputor types.
"""
    splitkwargs(::Type{T}, kwargs...) where T -> (imp, rem)

Takes a type with kwargs and returns the constructed type and the
unused kwargs which should be passed to the `impute!` call.

NOTE: This is only intended to be used internally
"""
function splitkwargs(::Type{T}, kwargs...) where T
    rem = Dict(kwargs...)
    kwdef = empty(rem)

    for f in fieldnames(T)
        if haskey(rem, f)
            kwdef[f] = rem[f]
            delete!(rem, f)
        end
    end

    return (T(; kwdef...), rem)
end

# Specialcase kwargs constructor for substitute.
# TODO: Add an imputor method that types should overwrite when necessary or have it fallback to `fieldnames`
function splitkwargs(::Type{Substitute}, kwargs...)
    rem = Dict(kwargs...)
    kwdef = empty(rem)

    for f in (:statistic, :robust, :weights)
        if haskey(rem, f)
            kwdef[f] = rem[f]
            delete!(rem, f)
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

for (f, v) in pairs(assertion_methods)
    typename = nameof(v)
    @eval begin
        function $f(data; kwargs...)
            a, rem = splitkwargs($typename, kwargs...)
            return assert(data, a; rem...)
        end
        function $f(; kwargs...)
            a, rem = splitkwargs($typename, kwargs...)
            return data -> assert(data, a; rem...)
        end
    end
end

for (f, v) in pairs(imputation_methods)
    typename = nameof(v)
    f! = Symbol(f, :!)

    @eval begin
        function $f(data; kwargs...)
            imp, rem = splitkwargs($typename, kwargs...)
            return impute(data, imp; rem...)
        end
        function $f!(data; kwargs...)
            imp, rem = splitkwargs($typename, kwargs...)
            return impute!(data, imp; rem...)
        end
        function $f(; kwargs...)
            imp, rem = splitkwargs($typename, kwargs...)
            return data -> impute(data, imp; rem...)
        end
        function $f!(; kwargs...)
            imp, rem = splitkwargs($typename, kwargs...)
            return data -> impute!(data, imp; rem...)
        end
    end
end

# Provide a specific functional API for Impute.Filter.
filter(data; kwargs...) = apply(data, Filter(); kwargs...)
filter!(data; kwargs...) = apply!(data, Filter(); kwargs...)
filter(; kwargs...) = data -> apply(data, Filter(); kwargs...)
filter!(; kwargs...) = data -> apply!(data, Filter(); kwargs...)
filter(f::Function, data; kwargs...) = apply(data, Filter(f); kwargs...)
filter!(f::Function, data; kwargs...) = apply!(data, Filter(f); kwargs...)
filter(f::Function; kwargs...) = data -> apply(data, Filter(f); kwargs...)
filter!(f::Function; kwargs...) = data -> apply!(data, Filter(f); kwargs...)

@doc """
    Impute.dropobs(data; dims=1)

[Deprecated] Removes missing observations from the `AbstractArray` or `Tables.table` provided.
See [DropObs](@ref) for details.

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

[Deprecated] Finds variables with missing values in a `AbstractMatrix` or `Tables.table` and
removes them from the input data. See [DropVars](@ref) for details.

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
