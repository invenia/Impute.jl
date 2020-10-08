module Impute

using BSON
using CSV
using DataDeps
using Distances
using IterTools
using Missings
using NamedDims
using NearestNeighbors
using Random
using Statistics
using StatsBase
using TableOperations
using Tables: Tables, materializer, istable

using Base.Iterators
using LinearAlgebra
using LinearAlgebra: Diagonal

#=
TODO List:
- [X] Drop Context
- [X] Introduce a separate Threshold type and `assert` function
- [X] Update old tests cases to use new type
- [X] Add deprecations for old `context` calls
- [X] Generalize the dimensionality behaviour using a `dims` keyword similar to the stats functions
  https://github.com/JuliaLang/Statistics.jl/blob/master/src/Statistics.jl#L164
- [X] ~Drop in-place calls?~
- [X] Base imputors should dispatch on `impute(AbstractArray{Union{T, Missing}}, imp)`
- [X] Replace `dropobs` and `dropvars` with `Impute.drop` and a `dims` keyword
- [X] Make `Chain` not an imputor and have it work on `Assertions` and `Imputors`
- [X] ~Add function for checking if a methods support some input data?~
- [X] Add more tests for NamedDims and AxisKeys
- [ ] Add walkthrough docs
=#

include("utils.jl")
include("assertions.jl")
include("imputors.jl")
include("filter.jl")
include("chain.jl")
include("deprecated.jl")
include("functional.jl")
include("data.jl")

__init__() = register_datadep()

end  # module
