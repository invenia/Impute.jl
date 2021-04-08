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


        @testset "Hashing" begin
            @test hash(tester.imp(tester.kwargs...)) == hash(tester.imp(tester.kwargs...))
        end

        @testset "Equality" begin
            @test tester.imp(tester.kwargs...) == tester.imp(tester.kwargs...)
        end

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

@testset "LimitedLOCF" begin
    @testset "Default" begin
        tester = ImputorTester(LimitedLOCF; max_gap_size=5)

        test_vector(tester)
        test_matrix(tester)
        test_dataframe(tester)
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

    types =  [
        ("Floats", allowmissing(1.0:1.0:20.0)),
        ("Ints", allowmissing(1:1:20)),
        ("Strings", allowmissing([randstring(4) for i in 1:20])),
    ]

    @testset "$t" for (t, array) in types
        a = copy(array)
        a[[2, 3, 7]] .= missing

        result = impute(a, LimitedLOCF(2))
        expected = copy(a)
        expected[2] = expected[1]
        expected[3] = expected[1]
        expected[7] = expected[6]

        @test result == expected

        # Test small gap size
        result = impute(a, LimitedLOCF(1))
        expected = copy(a)
        expected[7] = expected[6]

        @test isequal(result, expected)

        # Test with gap_axis
        distances = vcat(1:4, (5:20) * 5)

        result = impute(a, LimitedLOCF(3, distances))
        expected = copy(a)
        expected[2] = expected[1]
        expected[3] = expected[1]

        @test isequal(result, expected)

        @test_throws DimensionMismatch result = impute(a, LimitedLOCF(3, 1:10))

        # Test at endpoints
        b = copy(array)
        b[[1, 3, 20]] .= missing
        result = impute(b, LimitedLOCF(max_gap_size=2))
        @test ismissing(result[1])
        @test result[3] == b[2]
        @test result[20] == b[19]
    end
end

