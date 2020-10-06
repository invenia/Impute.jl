@testset "NOCB" begin
    @testset "Default" begin
        test_all(ImputorTester(NOCB))
    end

    @testset "Floats" begin
        a = allowmissing(1.0:1.0:20.0)
        a[[2, 3, 7]] .= missing

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

        # Test LOCF at endpoints
        b = ones(Union{Float64, Missing}, 20)
        b[[1, 3, 20]] .= missing
        result = Impute.nocb(b)
        @test result[1] == 1.0
        @test ismissing(result[20])
    end

    @testset "Ints" begin
        a = allowmissing(1:1:20)
        a[[2, 3, 7]] .= missing

        result = impute(a, NOCB())
        expected = copy(a)
        expected[2] = 4
        expected[3] = 4
        expected[7] = 8

        @test result == expected
    end

    @testset "Strings" begin
        a = allowmissing([randstring(4) for i in 1:20])
        a[[2, 3, 7]] .= missing

        result = impute(a, NOCB())
        expected = copy(a)
        expected[2] = expected[4]
        expected[3] = expected[4]
        expected[7] = expected[8]

        @test result == expected
    end
end
