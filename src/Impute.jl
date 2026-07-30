module Impute

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
include("declaremissings.jl")
include("imputors.jl")
include("filter.jl")
include("validators.jl")
include("chain.jl")
include("deprecated.jl")

include("functional.jl")

# Stub functions for extension - these will be defined when DataDeps, CSV, and BSON are loaded
function register_datadep end
function datasets end
function dataset end
function fetch_without_logs end
function load_csv end
function load_bson end

end  # module
