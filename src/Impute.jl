module Impute

using IterTools
using Statistics
using StatsBase
using Tables: Tables, materializer, istable

import Base.Iterators: drop

export impute, impute!, chain, chain!, drop, drop!, interp, interp!, ImputeError

function __init__()
    sym = join(["chain", "chain!", "drop", "drop!", "interp", "interp!"], ", ", " and ")

    @warn(
        """
        The following symbols will not be exported in future releases: $sym.
        Please qualify your calls with `Impute.<method>(...)` or explicitly import the symbol.
        """
    )

    @warn(
        """
        The default limit for all impute functions will be 1.0 going forward.
        If you depend on a specific threshold please pass in an appropriate `AbstractContext`.
        """
    )

    @warn(
        """
        All matrix imputation methods will be switching to the column-major convention
        (e.g., each column corresponds to an observation, and each row corresponds to a variable).
        To maintain the existing behaviour please pass `vardim=2` to the `Imputor` constructors
        or impute functions (e.g., `fill`, `interp`, `locf`).
        """
    )
end

"""
    ImputeError{T} <: Exception

Is thrown by `impute` methods when the limit of imputable values has been exceeded.

# Fields
* msg::T - the message to print.
"""
struct ImputeError{T} <: Exception
    msg::T
end

Base.showerror(io::IO, err::ImputeError) = println(io, "ImputeError: $(err.msg)")

include("context.jl")
include("imputors.jl")

const global imputation_methods = (
    drop = DropObs,
    dropobs = DropObs,
    dropvars = DropVars,
    interp = Interpolate,
    fill = Fill,
    locf = LOCF,
    nocb = NOCB,
)

include("deprecated.jl")

for (f, v) in pairs(imputation_methods)
    typename = nameof(v)
    f! = Symbol(f, :!)

    @eval begin
        $f(data; kwargs...) = impute($typename(; _extract_context_kwargs(kwargs...)...), data)
        $f!(data; kwargs...) = impute!($typename(; _extract_context_kwargs(kwargs...)...), data)
        $f(; kwargs...) = data -> impute($typename(; _extract_context_kwargs(kwargs...)...), data)
        $f!(; kwargs...) = data -> impute!($typename(; _extract_context_kwargs(kwargs...)...), data)
    end
end

@doc """
    Impute.dropobs(data; vardim=2, context=Context())

Removes missing observations from the `AbstractArray` or `Tables.table` provided.
See [DropObs](@ref) for details.

# Example
```
julia> using DataFrames; using Impute: Impute, Context

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.dropobs(df; vardim=1, context=Context(; limit=1.0))
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
    Impute.dropvars(data; vardim=2, context=Context())

Finds variables with too many missing values in a `AbstractMatrix` or `Tables.table` and
removes them from the input data. See [DropVars](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute, Context

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.dropvars(df; vardim=1, context=Context(; limit=0.2))
5×1 DataFrames.DataFrame
│ Row │ b        │
│     │ Float64⍰ │
├─────┼──────────┤
│ 1   │ 1.1      │
│ 2   │ 2.2      │
│ 3   │ 3.3      │
│ 4   │ missing  │
│ 5   │ 5.5      │
```
""" dropvars

@doc """
    Impute.interp(data; vardim=2, context=Context())

Performs linear interpolation between the nearest values in an vector.
See [Interpolate](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute, Context

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.interp(df; vardim=1, context=Context(; limit=1.0))
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 3.0      │ 3.3      │
│ 4   │ 4.0      │ 4.4      │
│ 5   │ 5.0      │ 5.5      │
```
""" interp

@doc """
    Impute.fill(data; value=mean, vardim=2, context=Context())

Fills in the missing data with a specific value. See [Fill](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute, Context

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.fill(df; value=-1.0, vardim=1, context=Context(; limit=1.0))
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ -1.0     │ 3.3      │
│ 4   │ -1.0     │ -1.0     │
│ 5   │ 5.0      │ 5.5      │
```
""" fill

@doc """
    Impute.locf(data; vardim=2, context=Context())

Iterates forwards through the `data` and fills missing data with the last existing
observation. See [LOCF](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute, Context

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.locf(df; vardim=1, context=Context(; limit=1.0))
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 2.0      │ 3.3      │
│ 4   │ 2.0      │ 3.3      │
│ 5   │ 5.0      │ 5.5      │
```
""" locf

@doc """
    Impute.nocb(data; vardim=2, context=Context())

Iterates backwards through the `data` and fills missing data with the next existing
observation. See [LOCF](@ref) for details.

# Example
```jldoctest
julia> using DataFrames; using Impute: Impute, Context

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.nocb(df; vardim=1, context=Context(; limit=1.0))
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 5.0      │ 3.3      │
│ 4   │ 5.0      │ 5.5      │
│ 5   │ 5.0      │ 5.5      │
```
""" nocb


end  # module
