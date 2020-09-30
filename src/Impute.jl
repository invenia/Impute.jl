module Impute

using Distances
using IterTools
using Missings
using NearestNeighbors
using Random
using Statistics
using StatsBase
using TableOperations
using Tables: Tables, materializer, istable

using Base.Iterators
using LinearAlgebra
using LinearAlgebra: Diagonal

import Base.Iterators: drop

#=
TODO List:
- [X] Drop Context
- [X] Introduce a separate Threshold type and `assert` function
- [X] Update old tests cases to use new type
- [X] Add deprecations for old `context` calls
- [ ] Generalize the dimensionality behaviour using a `dims` keyword similar to the stats functions
  https://github.com/JuliaLang/Statistics.jl/blob/master/src/Statistics.jl#L164
- [ ] Drop in-place calls
- [ ] Base imputors should dispatch on `impute(AbstractArray{Union{T, Missing}}, imp)`
- [ ] Replace `dropobs` and `dropvars` with `Impute.drop` and a `dims` keyword
- [ ] Make `Chain` not an imputor and have it work on `Assertions` and `Imputors`
- [ ] Add function for checking if a methods support some input data?
=#
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

include("utils.jl")
include("assertions.jl")
include("imputors.jl")
include("functional.jl")
include("deprecated.jl")

end  # module
