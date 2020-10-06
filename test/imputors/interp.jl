@testset "Interpolate" begin
    @testset "Default" begin
        test_all(ImputorTester(Interpolate))
    end

    @testset "Floats" begin
        # Defining our missing datasets
        a = allowmissing(1.0:1.0:20.0)
        a[[2, 3, 7]] .= missing

        result = impute(a, Interpolate())
        @test result == collect(1.0:1.0:20)
        @test result == Impute.interp(a)

        # Test in-place method
        a2 = copy(a)
        Impute.interp!(a2)
        @test a2 == result

        # Test interpolation between identical points
        b = ones(Union{Float64, Missing}, 20)
        b[[2, 3, 7]] .= missing
        @test Impute.interp(b) == ones(Union{Float64, Missing}, 20)

        # Test interpolation at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = Impute.interp(b)
        @test ismissing(result[1])
        @test ismissing(result[20])
    end

    @testset "Ints" begin
        # Defining our missing datasets
        a = allowmissing(1:1:20)
        a[[2, 3, 7]] .= missing

        result = impute(a, Interpolate())
        @test result == collect(1:1:20)
        @test result == Impute.interp(a)

        # Test in-place method
        a2 = copy(a)
        Impute.interp!(a2)
        @test a2 == result

        # Test interpolation between identical points
        b = ones(Union{Float64, Missing}, 20)
        b[[2, 3, 7]] .= missing
        @test Impute.interp(b) == ones(Union{Float64, Missing}, 20)

        # Test interpolation at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = Impute.interp(b)
        @test ismissing(result[1])
        @test ismissing(result[20])

        # Test inexact error
        # https://github.com/invenia/Impute.jl/issues/71
        c = [1, missing, 2, 3]
        @test_throws InexactError Impute.interp(c)
    end

    # TODO Test error cases on non-numeric types
end
