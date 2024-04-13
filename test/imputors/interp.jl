@testset "Interpolate" begin
    @testset "Default" begin
        tester = ImputorTester(Interpolate)

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

        # Test limiting
        c = allowmissing(1.0:1.0:20.0)
        c[13:15] .= missing

         # Limit too small for gap
        expected = copy(c)
        @test isequal(impute(c, Interpolate(; limit=2)), expected)

        # Limit matches gap size
        expected[13:15] .= [13.0, 14.0, 15.0]
        @test isequal(impute(c, Interpolate(; limit=3)), expected)
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

        # Test with UInt
        c = [0x1, missing, 0x3, 0x4]
        @test Impute.interp(c) == [0x1, 0x2, 0x3, 0x4]

        # Test reverse case where the increment is negative
        @test Impute.interp(reverse(c)) == [0x4, 0x3, 0x2, 0x1]

        # Test inexact error (no rounding mode provided)
        # https://github.com/invenia/Impute.jl/issues/71
        c = [1, missing, 2, 3]
        @test_throws InexactError Impute.interp(c)

        # Test with UInt
        c = [0x1, missing, 0x2, 0x3]
        @test_throws InexactError Impute.interp(c)

        # Test reverse case where the increment is negative
        @test_throws InexactError Impute.interp(reverse(c))

        # Test inexact cases with a rounding mode
        c = [1, missing, 2, 3]
        @test Impute.interp(c; r=RoundToZero) == [1, 1, 2, 3]

        # Test with UInt
        c = [0x1, missing, 0x2, 0x3]
        @test Impute.interp(c; r=RoundNearest) == [0x1, 0x1, 0x2, 0x3]

        # Test reverse case where the increment is negative
        @test Impute.interp(reverse(c); r=RoundUp) == [0x3, 0x2, 0x2, 0x1]

        # Test rounding doesn't cause values to exceed endpoint values
        @test Impute.interp([1, missing, missing, 2]; r=RoundUp) == [1, 2, 3, 2]
        @test Impute.interp([2, missing, missing, 1]; r=RoundUp) == [2, 2, 2, 1]
        @test Impute.interp([1, missing, missing, 0]; r=RoundDown) == [1, 0, -1, 0]
        @test_throws InexactError Impute.interp([0x1, missing, missing, 0x0]; r=RoundDown)
    end

    # TODO Test error cases on non-numeric types
end
