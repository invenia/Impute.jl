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
        result = impute(a, :drop; limit=0.2)
        expected = copy(a)
        deleteat!(expected, [2, 3, 7])

        @test result == expected
        @test result == drop(a)
    end

    @testset "Interpolate" begin
        result = impute(a, :interp; limit=0.2)
        @test result == collect(1.0:1.0:20)
        @test result == interp(a)
    end

    @testset "Fill" begin
        @testset "Value" begin
            fill_val = -1.0
            result = impute(a, :fill, fill_val; limit=0.2)
            expected = copy(a)
            expected[[2, 3, 7]] = fill_val

            @test result == expected
        end

        @testset "Mean" begin
            result = impute(a, :fill; limit=0.2)
            expected = copy(a)
            expected[[2, 3, 7]] = mean(drop(a))

            @test result == expected
        end
    end

    @testset "LOCF" begin
        result = impute(a, :locf; limit=0.2)
        expected = copy(a)
        expected[2] = 1.0
        expected[3] = 1.0
        expected[7] = 6.0

        @test result == expected
    end

    @testset "NOCB" begin
        result = impute(a, :nocb; limit=0.2)
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0

        @test result == expected
    end

    @testset "NullableArray" begin
        b = NullableArray(a)
        b[[2, 3, 7]] = Nullable()

        result = impute(b, :drop; limit=0.2)

        @test Array(result) == impute(a, :drop; limit=0.2)
    end

    @testset "DataArray" begin
        b = DataArray(a)
        b[[2, 3, 7]] = NA

        result = impute(b, :interp; limit=0.2)

        @test Array(result) == impute(a, :interp; limit=0.2)
    end

    @testset "DataFrame" begin
        data = dataset(DataFrames, "boot", "neuro")
        df = impute(data, :interp; limit=1.0)
    end

    @testset "DataTable" begin
        data = dataset(DataTables, "boot", "neuro")
        dt = impute(data, :drop; limit=1.0)
    end

    @testset "Matrix" begin
        data = hcat(dataset(DataTables, "boot", "neuro").columns...)

        @testset "Drop" begin
            result = drop(data)
            @test size(result, 1) == 4
        end

        @testset "Fill" begin
            result = impute(data, :fill, 0.0; limit=1.0)
            @test size(result) == size(data)
        end
    end

    @testset "Not enough data" begin
        @test_throws ImputeError impute(a, :drop)
    end

    @testset "Chain" begin
        data = hcat(dataset(DataTables, "boot", "neuro").columns...)
        result = chain(
            data,
            Impute.Interpolate(),
            Impute.LOCF(),
            Impute.NOCB();
            limit=1.0
        )

        @test size(result) == size(data)
        # Confirm that we don't have any more missing values
        @test !any(isnull, result)
    end

    @testset "Custom missing functions" begin
        data = dataset(DataTables, "boot", "neuro")
        @test impute(data, isnull, :drop; limit=1.0) == drop(data)
        result1 = chain(data, Impute.Interpolate(), Impute.Drop(); limit=1.0)
        result2 = chain(data, isnull, Impute.Interpolate(), Impute.Drop(); limit=1.0)
        @test result1 == result2
    end
end
