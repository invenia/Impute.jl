using AxisArrays
using AxisKeys
using CSV
using Combinatorics
using DataFrames
using Dates
using Distances
using Documenter
using LinearAlgebra
using Random
using Statistics
using StatsBase
using TableOperations
using Tables
using Test

using Impute
using Impute:
    Impute,
    Imputor,
    DropObs,
    DropVars,
    Interpolate,
    Fill,
    KNN,
    LOCF,
    NOCB,
    Replace,
    SRS,
    Standardize,
    Substitute,
    SVD,
    Filter,
    Threshold,
    apply,
    assert,
    impute,
    impute!,
    interp,
    chain,
    run,
    threshold


@testset "Impute" begin
    include("testutils.jl")

    include("assertions.jl")
    include("chain.jl")
    include("data.jl")
    include("deprecated.jl")
    include("filter.jl")
    include("imputors/interp.jl")
    include("imputors/knn.jl")
    include("imputors/locf.jl")
    include("imputors/nocb.jl")
    include("imputors/replace.jl")
    include("imputors/srs.jl")
    include("imputors/standardize.jl")
    include("imputors/substitute.jl")
    include("imputors/svd.jl")
    include("utils.jl")

    # Start running doctests before we wrap up technical changes and work
    # on more documentation
    doctest(Impute)
end
