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
