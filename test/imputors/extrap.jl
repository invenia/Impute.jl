@testset "Extrapolate" begin
    @testset "Floats" begin
        a = allowmissing(1.0:1.0:20.0)
        a[[1, 2, 3, 7, 13, 19, 20]] .= missing

        result = impute(a, Extrapolate())
        @test result[1:6] == collect(1.0:1.0:6.0)
        @test ismissing(result[7])
        @test result[14:20] == collect(14.0:1.0:20.0)
        @test ismissing(result[13])
        @test isequal(result, Impute.extrapolate(a))

        a2 = copy(a)
        Impute.extrapolate!(a2)
        @test isequal(a2, result)

        # Test reverse slope
        a = allowmissing(20.0:-1.0:1.0)
        a[[1, 2, 3, 7, 13, 19, 20]] .= missing

        result = impute(a, Extrapolate())
        @test result[1:6] == collect(20.0:-1.0:15.0)
        @test ismissing(result[7])
        @test result[14:20] == collect(7.0:-1.0:1.0)
        @test ismissing(result[13])
        @test isequal(result, Impute.extrapolate(a))

        a2 = copy(a)
        Impute.extrapolate!(a2)
        @test isequal(a2, result)

        # Test non-linear data
        a = allowmissing(log.(1.0:1.0:20.0))
        a[[1, 2, 3, 7, 13, 19, 20]] .= missing

        result = impute(a, Extrapolate())
        @test result[1:6] != collect(log.(1.0:1.0:6.0))
        @test ismissing(result[7])

        d = a[5] - a[4]
        @test result[1] == a[4] - (d * 3)
        @test result[2] == a[4] - (d * 2)
        @test result[3] == a[4] - (d * 1)


        @test result[14:20] != collect(log.(14.0:1.0:20.0))
        @test ismissing(result[13])
        d = a[18] - a[17]
        @test result[19] == a[18] + d
        @test result[20] == a[18] + (d * 2)
        @test isequal(result, Impute.extrapolate(a))

        # Test inconsistent steps
        a = allowmissing(1.0:1.0:20.0)
        a[[1, 2, 3, 7, 13, 19, 20]] .= missing
        a[4] = 4.5
        a[18] = 17.5
        result = impute(a, Extrapolate())

        @test result[1] == 3.0
        @test result[2] == 3.5
        @test result[3] == 4.0
        @test result[19] == 18.0
        @test result[20] == 18.5

        # Example with non-adjacent points
        a = allowmissing(1.0:1.0:20.0)
        a[[1, 2, 4, 7, 13, 18, 20]] .= missing
        result = impute(a, Extrapolate())

        @test result[1] == 1.0
        @test result[2] == 2.0
        @test result[20] == 20.0
    end

    @testset "Ints" begin
        a = allowmissing(1:1:20)
        a[[1, 2, 3, 7, 13, 19, 20]] .= missing

        result = impute(a, Extrapolate())
        @test result[1:6] == collect(1:1:6)
        @test ismissing(result[7])
        @test result[14:20] == collect(14:1:20)
        @test ismissing(result[13])
        @test isequal(result, Impute.extrapolate(a))

        a2 = copy(a)
        Impute.extrapolate!(a2)
        @test isequal(a2, result)

        # Example with non-adjacent points
        a = allowmissing(1:1:20)
        a[[1, 2, 4, 7, 13, 18, 20]] .= missing
        result = impute(a, Extrapolate())

        @test result[1] == 1
        @test result[2] == 2
        @test result[20] == 20

        # Example requiring rounding
        a = allowmissing(1:1:20)
        a[[1, 2, 4, 7, 13, 18, 20]] .= missing
        a[3] = 4
        a[19] = 20
        @test_throws InexactError impute(a, Extrapolate())

        result = impute(a, Extrapolate(; r=RoundUp))
        @test result[1] == 3
        @test result[2] == 4
        @test result[20] == 22

        # Example with gap size limit
        a = allowmissing(1:1:20)
        a[[1, 2, 3, 7, 13, 19, 20]] .= missing

        result = impute(a, Extrapolate(; limit=1))
        @test isequal(result, a)

        result = impute(a, Extrapolate(; limit=2))
        @test isequal(
            result[[1, 2, 3, 7, 13, 19, 20]],
            [missing, missing, missing, missing, missing, 19, 20]
        )

        # Example requiring rounding w/ Uints
        a = allowmissing(UInt.(1:1:20))
        a[[1, 2, 4, 7, 13, 18, 20]] .= missing
        a[3] = 4
        a[19] = 20

        result = impute(a, Extrapolate(; r=RoundUp))
        @test result[1] == 3
        @test result[2] == 4
        @test result[20] == 22

        # Example with gap size limit
        a = allowmissing(1:1:20)
        a[[1, 2, 3, 7, 13, 19, 20]] .= missing

        result = impute(a, Extrapolate(; limit=1))
        @test isequal(result, a)

        result = impute(a, Extrapolate(; limit=2))
        @test isequal(
            result[[1, 2, 3, 7, 13, 19, 20]],
            [missing, missing, missing, missing, missing, 19, 20]
        )
    end
end
