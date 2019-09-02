using Impute
using Tables
using Test
using AxisArrays
using DataFrames
using Dates
using RDatasets
using Statistics
using StatsBase
using Random

import Impute:
    Imputor,
    Drop,
    DropObs,
    DropVars,
    Interpolate,
    Fill,
    LOCF,
    NOCB,
    SRS,
    Context,
    WeightedContext,
    ImputeError


@testset "Impute" begin
    # Defining our missing datasets
    a = allowmissing(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)

    m = collect(reshape(a, 5, 4))

    table = DataFrame(
        :sin => allowmissing(sin.(1.0:1.0:20.0)),
        :cos => allowmissing(sin.(1.0:1.0:20.0)),
    )

    table.sin[[2, 3, 7, 12, 19]] .= missing

    include("testutils.jl")

    @testset "TestSuite: $T" for T in (DropObs, DropVars, Interpolate, Fill, LOCF, NOCB, SRS)
        test_all(ImputorTester(T))
    end

    @testset "Drop" begin
        @testset "DropObs" begin
            @testset "Vector" begin
                result = Impute.dropobs(a)
                expected = deleteat!(deepcopy(a), [2, 3, 7])
                @test result == expected
            end

            @testset "Matrix" begin
                result = Impute.dropobs(m)
                expected = m[[1, 4, 5], :]
                @test isequal(result, expected)
            end

            @testset "DataFrame" begin
                df = deepcopy(table)
                result = Impute.dropobs(df)
                expected = dropmissing(df)
                @test isequal(result, expected)
            end
        end

        @testset "DropVars" begin
            @testset "Vector" begin
                @test_throws MethodError Impute.dropvars(a)
            end

            @testset "Matrix" begin
                result = Impute.dropvars(m)
                expected = copy(m)[:, 3:4]
                @test isequal(result, expected)
            end

            @testset "DataFrame" begin
                df = deepcopy(table)
                result = Impute.dropvars(df)
                expected = select(df, :cos)

                @test isequal(result, expected)
            end
        end
    end

    @testset "Interpolate" begin
        @test interp(a) == collect(1.0:1.0:20)
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = Impute.fill(a; value=fill_val)
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val
            @test result == expected
        end

        @testset "Mean" begin
            result = Impute.fill(a; value=mean)
            expected = copy(a)
            expected[[2, 3, 7]] .= mean(a[mask])
            @test result == expected
        end
    end

    @testset "LOCF" begin
        result = Impute.locf(a)
        expected = copy(a)
        expected[[2, 3, 7]] = [1.0, 1.0, 6.0]
        @test result == expected
    end

    @testset "NOCB" begin
        result = Impute.nocb(a)
        expected = copy(a)
        expected[[2, 3, 7]] = [4.0, 4.0, 8.0]
        @test result == expected
    end

    @testset "SRS" begin
        result = Impute.srs(a; rng=MersenneTwister(137))
        expected = copy(a)
        expected[[2, 3, 7]] = [9.0, 16.0, 17.0]
        @test result == expected
    end

    @testset "Not enough data" begin
        ctx = Context(; limit=0.1)
        @test_throws ImputeError impute(a, DropObs(; context=ctx))
        @test_throws ImputeError Impute.dropobs(a; context=ctx)
    end

    # TODO: Flush out the Chain interface to better fit the TestSuite?
    @testset "Chain" begin
        orig = dataset("boot", "neuro")
        ctx = Context(; limit=1.0)

        @testset "DataFrame" begin
            result = Impute.interp(orig; context=ctx) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, Matrix(result))

            # We can also use the Chain type with explicit Imputor types
            result2 = impute(
                orig,
                Impute.Chain(
                    Impute.Interpolate(; context=ctx),
                    Impute.LOCF(),
                    Impute.NOCB()
                ),
            )

            # Test creating a Chain via Imputor composition
            imp = Impute.Interpolate(; context=ctx) ∘ Impute.LOCF() ∘ Impute.NOCB()
            result3 = impute(orig, imp)
            @test result == result2
            @test result == result3

            @testset "GroupedDataFrame" begin
                hod = repeat(1:24, 12 * 10)
                obj = repeat(1:12, 24 * 10)
                n = length(hod)

                df = DataFrame(
                    :hod => hod,
                    :obj => obj,
                    :val => allowmissing(
                        [sin(x) * cos(y) for (x, y) in zip(hod, obj)]
                    ),
                )

                df.val[rand(1:n, 20)] .= missing
                gdf1 = groupby(deepcopy(df), [:hod, :obj])
                gdf2 = groupby(df, [:hod, :obj])

                f1 = Impute.interp(; context=ctx) ∘ Impute.locf!() ∘ Impute.nocb!()
                f2 = Impute.interp!(; context=ctx) ∘ Impute.locf!() ∘ Impute.nocb!()

                result = mapreduce(f1, vcat, gdf1)
                @test df != result
                @test size(result) == (24 * 12 * 10, 3)
                @test all(!ismissing, Tables.matrix(result))

                # Test that we can also mutate the dataframe directly
                map(f2, gdf2)
                @test result == sort(df, (:hod, :obj))
            end
        end

        @testset "Column Table" begin
            result = Tables.columntable(orig) |>
                Impute.interp!(; context=ctx) |>
                Impute.locf!() |>
                Impute.nocb!() |>
                Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "Row Table" begin
            result = Tables.rowtable(orig) |>
                Impute.interp!(; context=ctx) |>
                Impute.locf!() |>
                Impute.nocb!() |>
                Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end

        @testset "Matrix" begin
            data = Matrix(orig)
            result = Impute.interp(data; context=ctx) |> Impute.locf!() |> Impute.nocb!()

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
            result = Impute.interp(data; context=ctx) |> Impute.locf!() |> Impute.nocb!()

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test all(!ismissing, result)
        end
    end

    @testset "Alternate missing functions" begin
        ctx1 = Context(; limit=1.0)
        ctx2 = Context(; limit=1.0, is_missing=isnan)
        data1 = dataset("boot", "neuro")                    # Missing values with `missing`
        data2 = Impute.fill(data1; value=NaN, context=ctx1)  # Missing values with `NaN`

        @test Impute.dropobs(data1; context=ctx1) == dropmissing(data1)

        result1 = Impute.interp(data1; context=ctx1) |> Impute.dropobs()
        result2 = Impute.interp(data2; context=ctx2) |> Impute.dropobs(; context=ctx2)

        @test result1 == result2
    end

    @testset "Contexts" begin
        @testset "Base" begin
            ctx = Context(; limit=0.1)
            @test_throws ImputeError Impute.dropobs(a; context=ctx)
            @test_throws ImputeError impute(a, DropObs(; context=ctx))
        end

        @testset "Weighted" begin
            # If we use an exponentially weighted context then we won't pass the limit
            # because missing earlier observations is less important than later ones.
            ctx = WeightedContext(eweights(20, 0.3); limit=0.1)
            @test isa(ctx, WeightedContext)
            result = impute(a, DropObs(; context=ctx))
            expected = copy(a)
            deleteat!(expected, [2, 3, 7])
            @test result == expected

            # If we reverse the weights such that earlier observations are more important
            # then our previous limit of 0.2 won't be enough to succeed.
            ctx = WeightedContext(reverse!(eweights(20, 0.3)); limit=0.2)
            @test_throws ImputeError impute(a, DropObs(; context=ctx))
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

    include("deprecated.jl")
end
