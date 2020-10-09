@testset "LOCF" begin
    @testset "Default" begin
        tester = ImputorTester(LOCF)

        test_hashing(tester)
        test_equality(tester)
        test_vector(tester)
        test_matrix(tester)
        # test_cube(tester)
        test_dataframe(tester)
        test_groupby(tester)
        test_axisarray(tester)
        test_nameddimsarray(tester)
        test_keyedarray(tester)
        test_columntable(tester)
        test_rowtable(tester)

        @testset "Cube" begin
            a = allowmissing(1.0:1.0:60.0)
            a[[2, 7, 18, 23, 34, 41, 55, 59, 60]] .= missing
            C = collect(reshape(a, 5, 4, 3))

            # Cube tests are expected to fail
            @test_throws MethodError impute(C, tester.imp(; tester.kwargs...); dims=3)
        end
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
