@testset "LOCF" begin
    @testset "Default" begin
        test_all(ImputorTester(LOCF))
    end

    @testset "Floats" begin
        a = allowmissing(1.0:1.0:20.0)
        a[[2, 3, 7]] .= missing

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

        # Test LOCF at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = Impute.locf(b)
        @test ismissing(result[1])
        @test result[20] == 1.0
    end

    @testset "Ints" begin
        a = allowmissing(1:1:20)
        a[[2, 3, 7]] .= missing

        result = impute(a, LOCF())
        expected = copy(a)
        expected[2] = 1
        expected[3] = 1
        expected[7] = 6

        @test result == expected
    end

    @testset "Strings" begin
        a = allowmissing([randstring(4) for i in 1:20])
        a[[2, 3, 7]] .= missing

        result = impute(a, LOCF())
        expected = copy(a)
        expected[2] = expected[1]
        expected[3] = expected[1]
        expected[7] = expected[6]

        @test result == expected
    end
end
