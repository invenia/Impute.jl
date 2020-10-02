# The standardize imputor is sufficiently different in its input and behaviour
# that we don't bother using the ImputorTester here.
@testset "Standardize" begin
    # List a couple known missing data values people might use.
    values = (NaN, 0.0, Nothing, "", 9999, -99, 0, DateTime(0))
    imp = Standardize(; values=values)

    @testset "Vector" begin
        @testset "disallowmissing" begin
            a = collect(1.0:1.0:20.0)
            a[[2, 3, 7]] .= [NaN, 0.0, NaN]

            result = impute(a, imp)
            @test eltype(result) == Union{Float64, Missing}
            @test all(ismissing, result[[2, 3, 7]])

            # In-place operation don't work when the source array doesn't allow missings.
            b = copy(a)
            result2 = impute!(b, imp)
            @test eltype(result2) == Float64
            @test isequal(result2[[2, 3, 7]], [NaN, 0.0, NaN])
        end

        @testset "allowmissing" begin
            a = allowmissing(collect(1.0:1.0:20.0))
            a[[2, 3, 7]] .= [NaN, 0.0, NaN]

            result = impute(a, imp)
            @test eltype(result) == Union{Float64, Missing}
            @test all(ismissing, result[[2, 3, 7]])

            # In-place operation don't work when the source array doesn't allow missings.
            b = copy(a)
            result2 = impute!(b, imp)
            @test eltype(result2) == Union{Float64, Missing}
            @test all(ismissing, result2[[2, 3, 7]])
        end

        @testset "All missing" begin
            # Test having only missing data
            c = fill(missing, 10)
            @test isequal(impute(c, imp), c)
        end
    end

    @testset "Matrix" begin
        @testset "disallowmissing" begin
            a = collect(1.0:1.0:20.0)
            a[[2, 3, 7]] .= [NaN, 0.0, NaN]
            m = collect(reshape(a, 5, 4))

            result = impute(m, imp)
            @test eltype(result) == Union{Float64, Missing}
            @test all(ismissing, result[[2, 3, 7]])

            # In-place operation don't work when the source array doesn't allow missings.
            n = copy(m)
            result2 = impute!(n, imp)
            @test eltype(result2) == Float64
            @test isequal(result2[[2, 3, 7]], [NaN, 0.0, NaN])
        end

        @testset "allowmissing" begin
            a = allowmissing(collect(1.0:1.0:20.0))
            a[[2, 3, 7]] .= [NaN, 0.0, NaN]
            m = collect(reshape(a, 5, 4))

            result = impute(m, imp)
            @test eltype(result) == Union{Float64, Missing}
            @test all(ismissing, result[[2, 3, 7]])

            # In-place operation don't work when the source array doesn't allow missings.
            n = copy(m)
            result2 = impute!(n, imp)
            @test eltype(result2) == Union{Float64, Missing}
            @test all(ismissing, result2[[2, 3, 7]])
        end

        @testset "All missing" begin
            # Test having only missing data
            c = fill(missing, 5, 4)
            @test isequal(impute(c, imp), c)
        end
    end
    @testset "Tables" begin
        @testset "DataFrame" begin
            table = DataFrame(
                :time => [DateTime(0), DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
                :loc => [12, -5, 9999],
                :age => [4, -99, 18],
                :val => [1.5, NaN, 3.0],
                :desc => ["foo", "bar", ""],
            )
            mtable = DataFrame(
                :time => allowmissing([DateTime(0), DateTime(2020, 02, 02), DateTime(2121, 12, 12)]),
                :loc => allowmissing([12, -5, 9999]),
                :age => allowmissing([4, -99, 18]),
                :val => allowmissing([1.5, NaN, 3.0]),
                :desc => allowmissing(["foo", "bar", ""]),
            )
            expected = DataFrame(
                :time => [missing, DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
                :loc => [12, -5, missing],
                :age => [4, missing, 18],
                :val => [1.5, missing, 3.0],
                :desc => ["foo", "bar", missing],
            )

            @testset "disallowmissing" begin
                result = impute(table, imp)
                @test isequal(result, expected)

                result2 = impute!(deepcopy(table), imp)
                @test !isequal(result2, expected)
            end

            @testset "allowmissing" begin
                result = impute(mtable, imp)
                @test isequal(result, expected)

                result2 = impute!(deepcopy(mtable), imp)
                @test isequal(result2, expected)
            end
        end

        @testset "Column Table" begin
            table = (
                time = [DateTime(0), DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
                loc = [12, -5, 9999],
                age = [4, -99, 18],
                val = [1.5, NaN, 3.0],
                desc = ["foo", "bar", ""],
            )
            mtable = (
                time = allowmissing([DateTime(0), DateTime(2020, 02, 02), DateTime(2121, 12, 12)]),
                loc = allowmissing([12, -5, 9999]),
                age = allowmissing([4, -99, 18]),
                val = allowmissing([1.5, NaN, 3.0]),
                desc = allowmissing(["foo", "bar", ""]),
            )
            expected = (
                time = [missing, DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
                loc = [12, -5, missing],
                age = [4, missing, 18],
                val = [1.5, missing, 3.0],
                desc = ["foo", "bar", missing],
            )

            @testset "disallowmissing" begin
                result = impute(table, imp)
                @test isequal(result, expected)

                result2 = impute!(deepcopy(table), imp)
                @test !isequal(result2, expected)
            end

            @testset "allowmissing" begin
                result = impute(mtable, imp)
                @test isequal(result, expected)

                result2 = impute!(deepcopy(mtable), imp)
                @test isequal(result2, expected)
            end
        end

        @testset "Row Table" begin
            table = Tables.rowtable((
                time = [DateTime(0), DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
                loc = [12, -5, 9999],
                age = [4, -99, 18],
                val = [1.5, NaN, 3.0],
                desc = ["foo", "bar", ""],
            ))
            mtable = Tables.rowtable((
                time = allowmissing([DateTime(0), DateTime(2020, 02, 02), DateTime(2121, 12, 12)]),
                loc = allowmissing([12, -5, 9999]),
                age = allowmissing([4, -99, 18]),
                val = allowmissing([1.5, NaN, 3.0]),
                desc = allowmissing(["foo", "bar", ""]),
            ))
            expected = Tables.rowtable((
                time = [missing, DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
                loc = [12, -5, missing],
                age = [4, missing, 18],
                val = [1.5, missing, 3.0],
                desc = ["foo", "bar", missing],
            ))

            @testset "disallowmissing" begin
                result = impute(table, imp)
                @test isequal(result, expected)

                result2 = impute!(deepcopy(table), imp)
                @test !isequal(result2, expected)
            end

            @testset "allowmissing" begin
                result = impute(mtable, imp)
                @test isequal(result, expected)

                result2 = impute!(deepcopy(mtable), imp)
                @test isequal(result2, expected)
            end
        end
    end
end
