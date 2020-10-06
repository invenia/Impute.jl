@testset "SRS" begin
    @testset "Default" begin
        tester = ImputorTester(SRS)
        test_hashing(tester)
        test_equality(tester)
        test_vector(tester)
        test_matrix(tester)
        test_dataframe(tester)
        # Behaviour is inconsistent for testing because of `rand` calls
        # test_groupby(tester)
        # test_axisarray(tester)
        # test_nameddimsarray(tester)
        # test_keyedarray(tester)
        test_columntable(tester)
        test_rowtable(tester)
    end

    @testset "Floats" begin
        a = allowmissing(1.0:1.0:20.0)
        a[[2, 3, 7]] .= missing
        result = impute(a, SRS(; rng=SequentialRNG()))
        expected = copy(a)
        expected[2] = 4.0
        expected[3] = 5.0
        expected[7] = 6.0

        @test result == expected

        @test result == Impute.srs(a; rng=SequentialRNG())

        a2 = copy(a)

        Impute.srs!(a2; rng=SequentialRNG())
        @test a2 == result
    end

    @testset "Ints" begin
        a = allowmissing(1:1:20)
        a[[2, 3, 7]] .= missing
        result = impute(a, SRS(; rng=SequentialRNG()))
        expected = copy(a)
        expected[2] = 4
        expected[3] = 5
        expected[7] = 6

        @test result == expected
    end

    @testset "Strings" begin
        a = allowmissing([randstring(4) for i in 1:20])
        a[[2, 3, 7]] .= missing
        result = impute(a, SRS(; rng=SequentialRNG()))
        expected = copy(a)
        expected[2] = expected[4]
        expected[3] = expected[5]
        expected[7] = expected[6]

        @test result == expected
    end
end
