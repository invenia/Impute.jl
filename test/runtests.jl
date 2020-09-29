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
    Threshold,
    ImputeError,
    assert,
    impute,
    impute!,
    interp,
    chain

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
        @testset "$T" for T in (DropObs, DropVars, Interpolate, Fill, LOCF, NOCB, SRS)
            @test T() == T()
        end
    end

    @testset "Drop" begin
        @testset "DropObs" begin
            @testset "Vector" begin
                result = impute(a, DropObs())
                expected = deleteat!(deepcopy(a), [2, 3, 7])

                @test result == expected
                @test result == Impute.dropobs(a)

                a2 = deepcopy(a)
                Impute.dropobs!(a2)
                @test a2 == expected
            end

            @testset "Matrix" begin
                result = impute(m, DropObs())
                expected = m[[1, 4, 5], :]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropobs(m))
                @test isequal(collect(result'), Impute.dropobs(collect(m'); dims=2))

                m_ = Impute.dropobs!(m)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(m, expected)
                @test isequal(m_, expected)
            end

            @testset "Tables" begin
                @testset "DataFrame" begin
                    df = deepcopy(table)
                    result = impute(df, DropObs())
                    expected = dropmissing(df)

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropobs(df))

                    df_ = Impute.dropobs!(df)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(df, expected)
                    @test isequal(df_, expected)
                end

                @testset "Column Table" begin
                    coltab = Tables.columntable(table)

                    result = impute(coltab, DropObs())
                    expected = Tables.columntable(dropmissing(table))

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropobs(coltab))

                    coltab_ = Impute.dropobs!(coltab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(coltab, expected)
                    @test isequal(coltab_, expected)
                end

                @testset "Row Table" begin
                    rowtab = Tables.rowtable(table)
                    result = impute(rowtab, DropObs())
                    expected = Tables.rowtable(dropmissing(table))

                    @show dropmissing(table)
                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropobs(rowtab))

                    rowtab_ = Impute.dropobs!(rowtab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    # @test_broken isequal(rowtab, expected)
                    @test isequal(rowtab_, expected)
                end
            end

            @testset "AxisArray" begin
                result = impute(aa, DropObs())
                expected = aa[[1, 4, 5], :]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropobs(aa))

                aa_ = Impute.dropobs!(aa)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(aa, expected)
                @test isequal(aa_, expected)
            end
        end

        @testset "DropVars" begin
            @testset "Vector" begin
                @test_throws MethodError Impute.dropvars(a)
            end

            @testset "Matrix" begin
                result = impute(m, DropVars())
                expected = copy(m)[:, 3:4]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropvars(m))
                @test isequal(collect(result'), Impute.dropvars(collect(m'); dims=2))

                m_ = Impute.dropvars!(m)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(m, expected)
                @test isequal(m_, expected)
            end

            @testset "Tables" begin
                @testset "DataFrame" begin
                    df = deepcopy(table)
                    result = impute(df, DropVars())
                    expected = select(df, :cos)

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropvars(df))

                    Impute.dropvars!(df)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(df, expected)
                end

                @testset "Column Table" begin
                    coltab = Tables.columntable(table)

                    result = impute(coltab, DropVars())
                    expected = Tables.columntable(TableOperations.select(coltab, :cos))

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropvars(coltab))

                    Impute.dropvars!(coltab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(coltab, expected)
                end

                @testset "Row Table" begin
                    rowtab = Tables.rowtable(table)
                    result = impute(rowtab, DropVars())
                    expected = Tables.rowtable(TableOperations.select(rowtab, :cos))

                    @test isequal(result, expected)
                    @test isequal(result, Impute.dropvars(rowtab))

                    Impute.dropvars!(rowtab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(rowtab, expected)
                end
            end
            @testset "AxisArray" begin
                result = impute(aa, DropVars())
                expected = copy(aa)[:, 3:4]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropvars(aa))

                aa_ = Impute.dropvars!(aa)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(aa, expected)
                @test isequal(aa_, expected)
            end
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
            expected = Matrix(Impute.dropobs(dataset("boot", "neuro")))
            data = Matrix(dataset("boot", "neuro"))

            result = impute(data, Fill(; value=0.0))
            @test size(result) == size(data)
            @test result == Impute.fill(data; value=0.0)

            data2 = copy(data)
            Impute.fill!(data2; value=0.0)
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

    @testset "Chain" begin
        orig = dataset("boot", "neuro")

        @testset "DataFrame" begin
            result = Impute.interp(orig) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, Matrix(result))

            # We can also use the Chain type with explicit Imputor types
            result2 = impute(
                orig,
                Impute.Chain(
                    Impute.Interpolate(),
                    Impute.LOCF(),
                    Impute.NOCB()
                ),
            )

            # Test creating a Chain via Imputor composition
            imp = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
            result3 = impute(orig, imp)
            @test result == result2
            @test result == result3

            @testset "GroupedDataFrame" begin
                T = NamedTuple{(:hod, :obj, :val), Tuple{Int, Int, Union{Float64, Missing}}}

                df = map(Iterators.product(1:24, 1:8, 0:19)) do t
                    hod, obj, x = t
                    # Deterministically return some `missing`s per hod/obj pair
                    return if x in (0, 5, 12, 19)
                        T((hod, obj, missing))
                    else
                        T((hod, obj, sin(hod) * cos(x) + obj))
                    end
                end |> DataFrame

                gdf1 = groupby(deepcopy(df), [:hod, :obj])
                gdf2 = groupby(df, [:hod, :obj])

                f1 = Impute.interp() ∘ Impute.locf!() ∘ Impute.nocb!()
                f2 = Impute.interp!() ∘ Impute.locf!() ∘ Impute.nocb!()

                result = mapreduce(f1, vcat, gdf1)
                # Check that the result isn't the same as the source dataframe
                @test df != result
                # Check that the size is still the same since we didn't drop any rows
                @test size(result) == size(df)
                # Check that there are no remaining missing values
                @test all(!ismissing, Tables.matrix(result))
                # Double check that our source dataframe still contains missings
                @test any(ismissing, Tables.matrix(df))

                # Test that we can also mutate the dataframe directly
                map(f2, gdf2)
                # Now we can check that we've replaced all the missing values in df
                @test all(!ismissing, Tables.matrix(df))
            end
        end

        @testset "Column Table" begin
            result = Tables.columntable(orig) |>
                Impute.interp!() |>
                Impute.locf!() |>
                Impute.nocb!() |>
                Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "Row Table" begin
            result = Tables.rowtable(orig) |>
                Impute.interp!() |>
                Impute.locf!() |>
                Impute.nocb!() |>
                Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "Matrix" begin
            data = Matrix(orig)
            result = Impute.interp(data) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "AxisArray" begin
            data = AxisArray(
                Matrix(orig),
                Axis{:row}(1:size(orig, 1)),
                Axis{:V}(names(orig)),
            )
            result = Impute.interp(data) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "KeyedArray" begin
            data = KeyedArray(Matrix(orig); row=1:size(orig, 1), V=names(orig))
            result = Impute.interp(data) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end
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
            t = Threshold(0.1)
            @test_throws AssertionError assert(a, t)
            @test_throws AssertionError assert(m, t)
            @test_throws AssertionError assert(aa, t)
            @test_throws AssertionError assert(table, t)

            t = Threshold(0.8)
            # Use isequal because we expect the results to contain missings
            @test isequal(assert(a, t), a)
            @test isequal(assert(m, t), m)
            @test isequal(assert(aa, t), aa)
            @test isequal(assert(table, t), table)
        end

        @testset "Weighted" begin
            # If we use an exponentially weighted context then we won't pass the limit
            # because missing earlier observations is less important than later ones.
            t = Threshold(0.8; weights=eweights(20, 0.3))
            @test isequal(assert(a, t), a)
            @test isequal(assert(table, t), table)

            t = Threshold(0.8; weights=eweights(5, 0.3))
            @test isequal(assert(m, t), m)
            @test isequal(assert(aa, t), aa)

            # If we reverse the weights such that earlier observations are more important
            # then our previous limit of 0.2 won't be enough to succeed.
            t = Threshold(0.1; weights=reverse!(eweights(20, 0.3)))
            @test_throws AssertionError assert(a, t)
            @test_throws AssertionError assert(table, t)

            t = Threshold(0.1; weights=reverse!(eweights(5, 0.3)))
            @test_throws AssertionError assert(m, t)
            @test_throws AssertionError assert(aa, t)

            @test_throws DimensionMismatch assert(a[1:10], t)
            @test_throws DimensionMismatch assert(m[1:3, :], t)
        end
    end

    @testset "Utils" begin
        M = [1.0 2.0 3.0 4.0 5.0; 1.1 2.2 3.3 4.4 5.5]

        @testset "obswise" begin
            @test map(sum, Impute.obswise(M; dims=2)) == [2.1, 4.2, 6.3, 8.4, 10.5]
            @test map(sum, Impute.obswise(M; dims=1)) == [15, 16.5]
        end

        @testset "varwise" begin
            @test map(sum, Impute.varwise(M; dims=2)) == [15, 16.5]
            @test map(sum, Impute.varwise(M; dims=1)) == [2.1, 4.2, 6.3, 8.4, 10.5]
        end

        @testset "filterobs" begin
            @test Impute.filterobs(x -> sum(x) > 5.0, M; dims=2) == M[:, 3:5]
            @test Impute.filterobs(x -> sum(x) > 15.0, M; dims=1) == M[[false, true], :]
        end

        @testset "filtervars" begin
            @test Impute.filtervars(x -> sum(x) > 15.0, M; dims=2) == M[[false, true], :]
            @test Impute.filtervars(x -> sum(x) > 5.0, M; dims=1) == M[:, 3:5]
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
                    knn_imputed = impute(copy(X), Impute.KNN(; k=2))
                    mean_imputed = impute(copy(X), Fill(; value=mean))

                    knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data, knn_imputed)) / i
                    mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data, mean_imputed)) / i
                end

                @test knn_nrmsd < mean_nrmsd
                # test type stability
                @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=2)))
                @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean)))
            end

            @testset "Iris - 0.25" begin
                X = add_missings(data, 0.25)

                knn_nrmsd, mean_nrmsd = 0.0, 0.0

                for i = 1:num_tests
                    knn_imputed = impute(copy(X), Impute.KNN(; k=2))
                    mean_imputed = impute(copy(X), Fill(; value=mean))

                    knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data, knn_imputed)) / i
                    mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data, mean_imputed)) / i
                end

                @test knn_nrmsd < mean_nrmsd
                # test type stability
                @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=2)))
                @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean)))
            end

            @testset "Iris - 0.35" begin
                X = add_missings(data, 0.35)

                knn_nrmsd, mean_nrmsd = 0.0, 0.0

                for i = 1:num_tests
                    knn_imputed = impute(copy(X), Impute.KNN(; k=2))
                    mean_imputed = impute(copy(X), Fill(; value=mean))

                    knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data, knn_imputed)) / i
                    mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data, mean_imputed)) / i
                end

                @test knn_nrmsd < mean_nrmsd
                # test type stability
                @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=2)))
                @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean)))
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
                knn_imputed = impute(copy(X), Impute.KNN(; k=2))
                mean_imputed = impute(copy(X), Fill(; value=mean))

                knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(data', knn_imputed)) / i
                mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(data', mean_imputed)) / i
            end

            @test knn_nrmsd < mean_nrmsd
            # test type stability
            @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=2)))
            @test typeof(X) == typeof(impute(copy(X), Fill(; value=mean)))
        end
    end

    include("deprecated.jl")
    include("testutils.jl")

    @testset "$T" for T in (DropObs, DropVars, Interpolate, Fill, LOCF, NOCB)
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

            svd_imputed = Impute.svd(X)
            mean_imputed = Impute.fill(copy(X))

            # With sufficient correlation between the variables and enough observation we
            # expect the svd imputation to perform severl times better than mean imputation.
            @test nrmsd(svd_imputed, data') < nrmsd(mean_imputed, data') * 0.5
        end

        # Test a case where we know SVD imputation won't perform well
        # (e.g., only a few variables, only )
        @testset "Data mismatch - too few variables" begin
            data = Matrix(dataset("Ecdat", "Electricity"))
            X = add_missings(data)

            svd_imputed = Impute.svd(X)
            mean_imputed = Impute.fill(copy(X))

            # If we don't have enough variables then SVD imputation will probably perform
            # about as well as mean imputation.
            @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
        end

        @testset "Data mismatch - poor low rank approximations" begin
            M = rand(100, 200)
            data = M * M'
            X = add_missings(data)

            svd_imputed = Impute.svd(X)
            mean_imputed = Impute.fill(copy(X))

            # If most of the variance in the original data can't be explained by a small
            # subset of the eigen values in the svd decomposition then our low rank approximations
            # won't perform very well.
            @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
        end
    end
end
