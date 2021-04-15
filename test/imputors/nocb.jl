@testset "NOCB" begin
    @testset "Default" begin
        tester = ImputorTester(NOCB)

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

        test_limited(tester)

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

        # Test limiting
        a[11:15] .= missing

        expected = copy(a)
        @test isequal(impute(a, NOCB(; limit=0)), expected)

        expected[2] = 4.0
        expected[3] = 4.0
        expected[7] = 8.0
        expected[13:15] .= 16.0

        @test isequal(impute(a, NOCB(; limit=3)), expected)
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
