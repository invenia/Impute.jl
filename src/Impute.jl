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

include("utils.jl")
include("imputors.jl")
include("filter.jl")
include("validators.jl")
include("chain.jl")
include("deprecated.jl")

include("functional.jl")
include("data.jl")

__init__() = register_datadep()

end  # module
