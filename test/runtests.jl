using Impute
using Base.Test

using NullableArrays
using DataArrays

import DataFrames
import DataTables

using RDatasets

@testset "Impute" begin
    a = collect(1.0:1.0:20.0)
    a[[2, 3, 7]] = NaN

    @testset "Drop" begin
        result = impute(a; method=:drop, limit=0.2)
        expected = copy(a)
        deleteat!(expected, [2, 3, 7])

        @test result == expected
        @test result == drop(a)
    end

    @testset "Interpolate" begin
        result = impute(a; limit=0.2)
        @test result == collect(1.0:1.0:20)
        @test result == interp(a)
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = impute(a; method=:fill, limit=0.2, value=fill_val)
            expected = copy(a)
            expected[[2, 3, 7]] = fill_val

            @test result == expected
        end

        @testset "Mean" begin
            result = impute(a; method=:fill, limit=0.2)
            expected = copy(a)
            expected[[2, 3, 7]] = mean(drop(a))

            @test result == expected
        end
    end

    @testset "LOCF" begin
        result = impute(a; method=:locf, limit=0.2)
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
    end

    @testset "NOCB" begin
        result = impute(a; method=:nocb, limit=0.2)
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
    end

    @testset "NullableArray" begin
        b = NullableArray(a)
        b[[2, 3, 7]] = Nullable()

        result = impute(b; limit=0.2)

        @test Array(result) == impute(a; limit=0.2)
    end

    @testset "DataArray" begin
        b = DataArray(a)
        b[[2, 3, 7]] = NA

        result = impute(b; limit=0.2)

        @test Array(result) == impute(a; limit=0.2)
    end

    @testset "DataFrame" begin
        data = dataset(DataFrames, "boot", "neuro")
        df = impute(data; limit=1.0)
    end

    @testset "DataTable" begin
        data = dataset(DataTables, "boot", "neuro")
        dt = impute(data; limit=1.0)
    end

    @testset "Not enough data" begin
        @test_throws ImputeError impute(a)
    end
end
