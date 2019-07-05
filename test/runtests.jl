using Impute
using Tables
using Test
using DataFrames
using RDatasets
using Statistics
using StatsBase

import Impute: Drop, Interpolate, Fill, LOCF, NOCB, Context, WeightedContext, ImputeError

@testset "Impute" begin
    a = Vector{Union{Float64, Missing}}(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)
    ctx = Context(; limit=0.2)

    @testset "Drop" begin
        result = impute(Drop(; context=ctx), a)
        expected = copy(a)
        deleteat!(expected, [2, 3, 7])

        @test result == expected
        @test result == Impute.drop(a; context=ctx)
    end

    @testset "Interpolate" begin
        result = impute(Interpolate(; context=ctx), a)
        @test result == collect(1.0:1.0:20)
        @test result == interp(a; context=ctx)

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
            result = impute(Fill(; value=fill_val, context=ctx), a)
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.fill(a; value=fill_val, context=ctx)
        end

        @testset "Mean" begin
            result = impute(Fill(; value=mean, context=ctx), a)
            expected = copy(a)
            expected[[2, 3, 7]] .= mean(a[mask])

            @test result == expected
            @test result == Impute.fill(a; value=mean, context=ctx)
        end
    end

    @testset "LOCF" begin
        result = impute(LOCF(; context=ctx), a)
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
        @test result == Impute.locf(a; context=ctx)
    end

    @testset "NOCB" begin
        result = impute(NOCB(; context=ctx), a)
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
        @test result == Impute.nocb(a; context=ctx)
    end

    @testset "DataFrame" begin
        ctx = Context(; limit=1.0)
        data = dataset("boot", "neuro")
        df = impute(Interpolate(; context=ctx), data)
    end

    @testset "Matrix" begin
        ctx = Context(; limit=1.0)
        data = Matrix(dataset("boot", "neuro"))

        @testset "Drop" begin
            result = impute(Drop(; context=ctx), data)
            @test size(result, 1) == 4
            @test result == Impute.drop(data; context=ctx)
        end

        @testset "Fill" begin
            result = impute(Fill(; value=0.0, context=ctx), data)
            @test size(result) == size(data)
            @test result == Impute.fill(data; value=0.0, context=ctx)
        end
    end

    @testset "Not enough data" begin
        ctx = Context(; limit=0.1)
        @test_throws ImputeError impute(Drop(; context=ctx), a)
        @test_throws ImputeError Impute.drop(a; context=ctx)
    end

    @testset "Chain" begin
        orig = dataset("boot", "neuro")
        ctx = Context(; limit=1.0)

        @testset "DataFrame" begin
            result = Impute.interp(orig; context=ctx) |> Impute.locf() |> Impute.nocb()

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test !any(ismissing, Matrix(result))
        end

        @testset "Column Table" begin
            result = Tables.columntable(orig) |>
                Impute.interp(; context=ctx) |>
                Impute.locf() |>
                Impute.nocb() |>
                Tables.matrix

            @test size(result) == size(orig)
            # Confirm that we don't have any more missing values
            @test !any(ismissing, result)
        end

        @testset "Matrix" begin
            data = Matrix(orig)
            result = Impute.interp(data; context=ctx) |> Impute.locf() |> Impute.nocb()

            @test size(result) == size(data)
            # Confirm that we don't have any more missing values
            @test !any(ismissing, result)
        end
    end

    @testset "Alternate missing functions" begin
        ctx1 = Context(; limit=1.0)
        ctx2 = Context(; limit=1.0, is_missing=isnan)
        data1 = dataset("boot", "neuro")                    # Missing values with `missing`
        data2 = Impute.fill(data1; value=NaN, context=ctx1)  # Missing values with `NaN`

        @test Impute.drop(data1; context=ctx1) == dropmissing(data1)

        result1 = Impute.interp(data1; context=ctx1) |> Impute.drop()
        result2 = Impute.interp(data2; context=ctx2) |> Impute.drop(; context=ctx2)

        @test result1 == result2
    end

    @testset "Contexts" begin
        @testset "Base" begin
            ctx = Context(; limit=0.1)
            @test_throws ImputeError Impute.drop(a; context=ctx)
            @test_throws ImputeError impute(Drop(; context=ctx), a)
        end

        @testset "Weighted" begin
            # If we use an exponentially weighted context then we won't pass the limit
            # because missing earlier observations is less important than later ones.
            ctx = WeightedContext(eweights(20, 0.3); limit=0.1)
            @test isa(ctx, WeightedContext)
            result = impute(Drop(), ctx, a)
            expected = copy(a)
            deleteat!(expected, [2, 3, 7])
            @test result == expected

            # If we reverse the weights such that earlier observations are more important
            # then our previous limit of 0.2 won't be enough to succeed.
            ctx = WeightedContext(reverse!(eweights(20, 0.3)); limit=0.2)
            @test_throws ImputeError impute(Drop(), ctx, a)
        end
    end

    include("deprecated.jl")
end
