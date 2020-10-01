using AxisArrays
using AxisKeys
using Combinatorics
using DataFrames
using Dates
using Distances
using LinearAlgebra
using RDatasets
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
    LOCF,
    NOCB,
    SRS,
    Filter,
    Threshold,
    ImputeError,
    apply,
    assert,
    impute,
    impute!,
    interp,
    chain,
    run,
    threshold

function add_missings(X, ratio=0.1)
    result = Matrix{Union{Float64, Missing}}(X)

    for i in 1:floor(Int, length(X) * ratio)
        result[rand(1:length(X))] = missing
    end

    return result
end

function add_missings_single(X, ratio=0.1)
    result = Matrix{Union{Float64, Missing}}(X)

    randcols = 1:floor(Int, size(X, 2) * ratio)
    for col in randcols
        result[rand(1:size(X, 1)), col] = missing
    end

    return result
end

# A sequential RNG for consistent testing across julia versions
mutable struct SequentialRNG <: AbstractRNG
    idx::Int
end
SequentialRNG(; start_idx=1) = SequentialRNG(start_idx)

function Base.rand(srng::SequentialRNG, x::Vector)
    srng.idx = srng.idx < length(x) ? srng.idx + 1 : 1
    return x[srng.idx]
end

@testset "Impute" begin
    # Defining our missing datasets
    a = allowmissing(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)

    # We call collect to not have a wrapper type that references the same data.
    m = collect(reshape(a, 5, 4))

    aa = AxisArray(
        deepcopy(m),
        Axis{:time}(DateTime(2017, 6, 5, 5):Hour(1):DateTime(2017, 6, 5, 9)),
        Axis{:id}(1:4)
    )

    table = DataFrame(
        :sin => allowmissing(sin.(1.0:1.0:20.0)),
        :cos => allowmissing(sin.(1.0:1.0:20.0)),
    )

    table.sin[[2, 3, 7, 12, 19]] .= missing

    @testset "Equality" begin
        @testset "$T" for T in (Interpolate, Fill, LOCF, NOCB, SRS)
            @test T() == T()
        end
    end

    @testset "Interpolate" begin
        result = impute(a, Interpolate())
        @test result == collect(1.0:1.0:20)
        @test result == interp(a)

        # Test in-place method
        a2 = copy(a)
        Impute.interp!(a2)
        @test a2 == result

        # Test interpolation between identical points
        b = ones(Union{Float64, Missing}, 20)
        b[[2, 3, 7]] .= missing
        @test interp(b) == ones(Union{Float64, Missing}, 20)

        # Test interpolation at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = interp(b)
        @test ismissing(result[1])
        @test ismissing(result[20])
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = impute(a, Fill(; value=fill_val))
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.fill(a; value=fill_val)
        end

        @testset "Mean" begin
            result = impute(a, Fill(; value=mean))
            expected = copy(a)
            expected[[2, 3, 7]] .= mean(a[mask])

            @test result == expected
            @test result == Impute.fill(a; value=mean)

            a2 = copy(a)
            Impute.fill!(a2)
            @test a2 == result
        end

        @testset "Matrix" begin
            data = Matrix(dataset("boot", "neuro"))

            result = impute(data, Fill(; value=0.0); dims=:cols)
            @test size(result) == size(data)
            @test result == Impute.fill(data; value=0.0, dims=:cols)

            data2 = copy(data)
            Impute.fill!(data2; value=0.0, dims=:cols)
            @test data2 == result
        end
    end

    @testset "LOCF" begin
        result = impute(a, LOCF())
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
        @test result == Impute.locf(a)

        a2 = copy(a)
        Impute.locf!(a2)
        @test a2 == result
    end

    @testset "NOCB" begin
        result = impute(a, NOCB())
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
        @test result == Impute.nocb(a)

        a2 = copy(a)
        Impute.nocb!(a2)
        @test a2 == result
    end

    @testset "SRS" begin
        result = impute(a, SRS(; rng=SequentialRNG()))
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 5.0
        expected[7] = 6.0

        @test result == expected

        @test result == Impute.srs(a; rng=SequentialRNG())

        a2 = copy(a)

        Impute.srs!(a2; rng=SequentialRNG())
        @test a2 == result
    end

    # TODO: Replace with example using `Missings.replace` maybe with an extension for impute?
    # @testset "Alternate missing functions" begin
    #     ctx1 = Context(; limit=1.0)
    #     ctx2 = Context(; limit=1.0, is_missing=isnan)
    #     data1 = dataset("boot", "neuro")                    # Missing values with `missing`
    #     data2 = Impute.fill(data1; value=NaN, context=ctx1)  # Missing values with `NaN`

    #     @test Impute.dropobs(data1; context=ctx1) == dropmissing(data1)

    #     result1 = Impute.interp(data1; context=ctx1) |> Impute.dropobs()
    #     result2 = Impute.interp(data2; context=ctx2) |> Impute.dropobs(; context=ctx2)

    #     @test result1 == result2
    # end

    @testset "Assertions" begin
        @testset "Base" begin
            t = Threshold(; ratio=0.1)
            @test_throws AssertionError assert(a, t)
            @test_throws AssertionError assert(m, t)
            @test_throws AssertionError assert(aa, t)
            @test_throws AssertionError assert(table, t)

            t = Threshold(; ratio=0.8)
            # Use isequal because we expect the results to contain missings
            @test isequal(assert(a, t), a)
            @test isequal(assert(m, t), m)
            @test isequal(assert(aa, t), aa)
            @test isequal(assert(table, t), table)
        end

        @testset "Weighted" begin
            # If we use an exponentially weighted context then we won't pass the limit
            # because missing earlier observations is less important than later ones.
            t = Threshold(; ratio=0.8, weights=eweights(20, 0.3))
            @test isequal(assert(a, t), a)
            @test isequal(assert(table, t), table)

            @test isequal(threshold(m; ratio=0.8, weights=eweights(5, 0.3)), m)
            @test isequal(threshold(m; ratio=0.8, weights=eweights(5, 0.3)), aa)

            # If we reverse the weights such that earlier observations are more important
            # then our previous limit of 0.2 won't be enough to succeed.
            t = Threshold(; ratio=0.1, weights=reverse!(eweights(20, 0.3)))
            @test_throws AssertionError assert(a, t)
            @test_throws AssertionError assert(table, t)

            t = Threshold(; ratio=0.1, weights=reverse!(eweights(5, 0.3)))
            @test_throws AssertionError assert(m, t)
            @test_throws AssertionError assert(aa, t)

            @test_throws DimensionMismatch assert(a[1:10], t)
            @test_throws DimensionMismatch assert(m[1:3, :], t)
        end
    end

    @testset "KNN" begin
        @testset "Iris" begin
            # Reference
            # P. Schimitt, et. al
            # A comparison of six methods for missing data imputation
            iris = dataset("datasets", "iris")
            iris2 = filter(row -> row[:Species] == "versicolor" || row[:Species] == "virginica", iris)
            data = Array(iris2[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
            num_tests = 100

            @testset "Iris - 0.15" begin
                X = add_missings(data, 0.15)

                knn_nrmsd, mean_nrmsd = 0.0, 0.0

                for i = 1:num_tests
                    knn_imputed = impute(copy(X), Impute.KNN(; k=2); dims=:cols)
                    mean_imputed = impute(copy(X), Fill(; value=mean); dims=:cols)

                    knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data, knn_imputed)) / i
                    mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data, mean_imputed)) / i
                end

                @test knn_nrmsd < mean_nrmsd
                # test type stability
                @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=2); dims=:cols))
                @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean); dims=:cols))
            end

            @testset "Iris - 0.25" begin
                X = add_missings(data, 0.25)

                knn_nrmsd, mean_nrmsd = 0.0, 0.0

                for i = 1:num_tests
                    knn_imputed = impute(copy(X), Impute.KNN(; k=2); dims=:cols)
                    mean_imputed = impute(copy(X), Fill(; value=mean); dims=:cols)

                    knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data, knn_imputed)) / i
                    mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data, mean_imputed)) / i
                end

                @test knn_nrmsd < mean_nrmsd
                # test type stability
                @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=2); dims=:cols))
                @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean); dims=:cols))
            end

            @testset "Iris - 0.35" begin
                X = add_missings(data, 0.35)

                knn_nrmsd, mean_nrmsd = 0.0, 0.0

                for i = 1:num_tests
                    knn_imputed = impute(copy(X), Impute.KNN(; k=2); dims=:cols)
                    mean_imputed = impute(copy(X), Fill(; value=mean); dims=:cols)

                    knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data, knn_imputed)) / i
                    mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data, mean_imputed)) / i
                end

                @test knn_nrmsd < mean_nrmsd
                # test type stability
                @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=2); dims=:cols))
                @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean); dims=:cols))
            end
        end

        # Test a case where we expect kNN to perform well (e.g., many variables, )
        @testset "Data match" begin
            data = mapreduce(hcat, 1:1000) do i
                seeds = [sin(i), cos(i), tan(i), atan(i)]
                mapreduce(vcat, combinations(seeds)) do args
                    [
                        +(args...),
                        *(args...),
                        +(args...) * 100,
                        +(abs.(args)...),
                        (+(args...) * 10) ^ 2,
                        (+(abs.(args)...) * 10) ^ 2,
                        log(+(abs.(args)...) * 100),
                        +(args...) * 100 + rand(-10:0.1:10),
                    ]
                end
            end

            X = add_missings(data')
            num_tests = 100

            knn_nrmsd, mean_nrmsd = 0.0, 0.0

            for i = 1:num_tests
                knn_imputed = impute(copy(X), Impute.KNN(; k=4); dims=:cols)
                mean_imputed = impute(copy(X), Fill(; value=mean); dims=:cols)

                knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data', knn_imputed)) / i
                mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data', mean_imputed)) / i
            end

            @test knn_nrmsd < mean_nrmsd
            # test type stability
            @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=4); dims=:cols))
            @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean); dims=:cols))
        end
    end

    include("deprecated.jl")
    include("filter.jl")
    include("chain.jl")
    include("testutils.jl")

    @testset "$T" for T in (Interpolate, Fill, LOCF, NOCB)
        test_all(ImputorTester(T))
    end

    @testset "SVD" begin
        # Test a case where we expect SVD to perform well (e.g., many variables, )
        @testset "Data match" begin
            data = mapreduce(hcat, 1:1000) do i
                seeds = [sin(i), cos(i), tan(i), atan(i)]
                mapreduce(vcat, combinations(seeds)) do args
                    [
                        +(args...),
                        *(args...),
                        +(args...) * 100,
                        +(abs.(args)...),
                        (+(args...) * 10) ^ 2,
                        (+(abs.(args)...) * 10) ^ 2,
                        log(+(abs.(args)...) * 100),
                        +(args...) * 100 + rand(-10:0.1:10),
                    ]
                end
            end

            # println(svd(data').S)
            X = add_missings(data')

            svd_imputed = Impute.svd(X; dims=:cols)
            mean_imputed = Impute.fill(copy(X); dims=:cols)

            # With sufficient correlation between the variables and enough observation we
            # expect the svd imputation to perform severl times better than mean imputation.
            @test nrmsd(svd_imputed, data') < nrmsd(mean_imputed, data') * 0.5
        end

        # Test a case where we know SVD imputation won't perform well
        # (e.g., only a few variables, only )
        @testset "Data mismatch - too few variables" begin
            data = Matrix(dataset("Ecdat", "Electricity"))
            X = add_missings(data)

            svd_imputed = Impute.svd(X; dims=:cols)
            mean_imputed = Impute.fill(copy(X); dims=:cols)

            # If we don't have enough variables then SVD imputation will probably perform
            # about as well as mean imputation.
            @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
        end

        @testset "Data mismatch - poor low rank approximations" begin
            M = rand(100, 200)
            data = M * M'
            X = add_missings(data)

            svd_imputed = Impute.svd(X; dims=:cols)
            mean_imputed = Impute.fill(copy(X); dims=:cols)

            # If most of the variance in the original data can't be explained by a small
            # subset of the eigen values in the svd decomposition then our low rank approximations
            # won't perform very well.
            @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
        end
    end
end
