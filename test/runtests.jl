using Impute
using Tables
using Test
using DataFrames
using RDatasets
using Statistics
using StatsBase
using Random

import Impute:
    Drop,
    DropObs,
    DropVars,
    Interpolate,
    Fill,
    LOCF,
    NOCB,
    HotDeck,
    Context,
    WeightedContext,
    ImputeError

@testset "Impute" begin
    a = allowmissing(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)
    ctx = Context(; limit=0.2)

    @testset "Equality $T" for T in (DropObs, DropVars, Interpolate, Fill, LOCF, NOCB, HotDeck)
        @test T() == T()
    end

    @testset "Drop" begin
        @testset "DropObs" begin
            result = impute(a, DropObs(; context=ctx))
            expected = copy(a)
            deleteat!(expected, [2, 3, 7])

            @test result == expected
            @test result == Impute.dropobs(a; context=ctx)

            a2 = copy(a)
            Impute.dropobs!(a2; context=ctx)
            @test a2 == expected
        end
        @testset "DropVars" begin
            @testset "Vector" begin
                @test_throws MethodError Impute.dropvars(a)
            end

            @testset "Matrix" begin
                m = reshape(a, 5, 4)

                result = impute(m, DropVars(; context=ctx))
                expected = copy(m)[:, 2:4]

                @test isequal(result, expected)
                @test isequal(result, Impute.dropvars(m; context=ctx))
                @test isequal(result', Impute.dropvars(m'; vardim=1, context=ctx))

                Impute.dropvars!(m; context=ctx)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(m, expected)
            end
            @testset "DataFrame" begin
                df = DataFrame(
                    :sin => allowmissing(sin.(1.0:1.0:20.0)),
                    :cos => allowmissing(sin.(1.0:1.0:20.0)),
                )
                df.sin[[2, 3, 7, 12, 19]] .= missing
                df.cos[[4, 9]] .= missing

                result = impute(df, DropVars(; context=ctx))
                expected = select(df, :cos)

                @test isequal(result, expected)
                @test isequal(result, Impute.dropvars(df; context=ctx))

                Impute.dropvars!(df; context=ctx)
                # The mutating test is broken because we need to making a copy of
                # the original table
                @test_broken isequal(df, expected)
            end
        end
    end

    @testset "Interpolate" begin
        result = impute(a, Interpolate(; context=ctx))
        @test result == collect(1.0:1.0:20)
        @test result == interp(a; context=ctx)

        # Test in-place method
        a2 = copy(a)
        Impute.interp!(a2; context=ctx)
        @test a2 == result

        # Test interpolation between identical points
        b = ones(Union{Float64, Missing}, 20)
        b[[2, 3, 7]] .= missing
        @test interp(b; context=ctx) == ones(Union{Float64, Missing}, 20)

        # Test interpolation at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = interp(b; context=ctx)
        @test ismissing(result[1])
        @test ismissing(result[20])
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = impute(a, Fill(; value=fill_val, context=ctx))
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.fill(a; value=fill_val, context=ctx)
        end

        @testset "Mean" begin
            result = impute(a, Fill(; value=mean, context=ctx))
            expected = copy(a)
            expected[[2, 3, 7]] .= mean(a[mask])

            @test result == expected
            @test result == Impute.fill(a; value=mean, context=ctx)

            a2 = copy(a)
            Impute.fill!(a2; context=ctx)
            @test a2 == result
        end
    end

    @testset "LOCF" begin
        result = impute(a, LOCF(; context=ctx))
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
        @test result == Impute.locf(a; context=ctx)

        a2 = copy(a)
        Impute.locf!(a2; context=ctx)
        @test a2 == result
    end

    @testset "NOCB" begin
        result = impute(a, NOCB(; context=ctx))
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
        @test result == Impute.nocb(a; context=ctx)

        a2 = copy(a)
        Impute.nocb!(a2; context=ctx)
        @test a2 == result
    end

    @testset "HotDeck" begin
        # Setting seed because Hot Deck isn't deterministic
        Random.seed!(137)
        result = impute(a, HotDeck(; context=ctx))
        expected = copy(a)
        expected[2] = 9.0
        expected[3] = 16.0
        expected[7] = 17.0

        @test result == expected

        Random.seed!(137)
        @test result == Impute.hotdeck(a; context=ctx)

        a2 = copy(a)

        Random.seed!(137)
        Impute.hotdeck!(a2; context=ctx)
        @test a2 == result
    end


    @testset "DataFrame" begin
        ctx = Context(; limit=1.0)
        @testset "Single DataFrame" begin
            data = dataset("boot", "neuro")
            df = impute(data, Interpolate(; context=ctx))
            @test isequal(df, Impute.interp(data; context=ctx))
        end
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

            result = vcat(f1.(gdf1)...)
            @test df != result
            @test size(result) == (24 * 12 * 10, 3)
            @test all(!ismissing, Tables.matrix(result))

            # Test that we can also mutate the dataframe directly
            f2.(gdf2)
            @test result == sort(df, (:hod, :obj))
        end
    end

    @testset "Matrix" begin
        ctx = Context(; limit=1.0)
        expected = Matrix(Impute.dropobs(dataset("boot", "neuro"); context=ctx))
        data = Matrix(dataset("boot", "neuro"))

        @testset "Drop" begin
            result = impute(data, DropObs(; context=ctx))
            @test size(result, 1) == 4
            @test result == Impute.dropobs(data; context=ctx)

            @test result == expected
            @test Impute.dropobs(data'; vardim=1, context=ctx) == expected'
        end

        @testset "Fill" begin
            result = impute(data, Fill(; value=0.0, context=ctx))
            @test size(result) == size(data)
            @test result == Impute.fill(data; value=0.0, context=ctx)

            data2 = copy(data)
            Impute.fill!(data2; value=0.0, context=ctx)
            @test data2 == result
        end
    end

    @testset "Not enough data" begin
        ctx = Context(; limit=0.1)
        @test_throws ImputeError impute(a, DropObs(; context=ctx))
        @test_throws ImputeError Impute.dropobs(a; context=ctx)
    end

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

            @test result == result2
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

        @testset "Matrix" begin
            data = Matrix(orig)
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

        result1 = Impute.interp(data1; context=ctx1) |> Impute.dropobs!()
        result2 = Impute.interp(data2; context=ctx2) |> Impute.dropobs!(; context=ctx2)

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
        drop_dim1 = DropObs(; vardim=1)
        drop_dim2 = DropObs(; vardim=2)
        M = [1.0 2.0 3.0 4.0 5.0; 1.1 2.2 3.3 4.4 5.5]

        @testset "obswise" begin
            @test map(sum, Impute.obswise(drop_dim1, M)) == [2.1, 4.2, 6.3, 8.4, 10.5]
            @test map(sum, Impute.obswise(drop_dim2, M)) == [15, 16.5]
        end

        @testset "varwise" begin
            @test map(sum, Impute.varwise(drop_dim1, M)) == [15, 16.5]
            @test map(sum, Impute.varwise(drop_dim2, M)) == [2.1, 4.2, 6.3, 8.4, 10.5]
        end

        @testset "filterobs" begin
            @test Impute.filterobs(x -> sum(x) > 5.0, drop_dim1, M) == M[:, 3:5]
            @test Impute.filterobs(x -> sum(x) > 15.0, drop_dim2, M) == M[[false, true], :]
        end

        @testset "filtervars" begin
            @test Impute.filtervars(x -> sum(x) > 15.0, drop_dim1, M) == M[[false, true], :]
            @test Impute.filtervars(x -> sum(x) > 5.0, drop_dim2, M) == M[:, 3:5]
        end
    end

    include("deprecated.jl")
end
