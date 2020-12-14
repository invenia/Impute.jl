using AxisArrays
using AxisKeys
using CSV
using Combinatorics
using DataFrames
using Dates
using Distances
using Documenter
using HypothesisTests
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
    Chain,
    DropObs,
    DropVars,
    Interpolate,
    Fill,
    KNN,
    LOCF,
    NOCB,
    Replace,
    SRS,
    DeclareMissings,
    Substitute,
    WeightedSubstitute,
    SVD,
    Filter,
    Threshold,
    WeightedThreshold,
    ThresholdError,
    apply,
    apply!,
    impute,
    impute!,
    interp,
    run,
    threshold,
    wthreshold,
    validate


@testset "Impute" begin
    include("testutils.jl")

    include("validators.jl")
    include("declaremissings.jl")
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
    include("imputors/substitute.jl")
    include("imputors/svd.jl")
    include("utils.jl")

    # Start running doctests before we wrap up technical changes and work
    # on more documentation
    doctest(Impute)
end
