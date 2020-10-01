@testset "Filter" begin
    # Defining our missing datasets
    a = allowmissing(1.0:1.0:20.0)
    a[[2, 3, 7]] .= missing
    mask = map(!ismissing, a)

    # We call collect to not have a wrapper type that references the same data.
    m = collect(reshape(a, 5, 4))

    aa = AxisArray(
        deepcopy(m),
        Axis{:time}(DateTime(2017, 6, 5, 5):Hour(1):DateTime(2017, 6, 5, 9)),
        Axis{:id}(1:4)
    )

    table = DataFrame(
        :sin => allowmissing(sin.(1.0:1.0:20.0)),
        :cos => allowmissing(sin.(1.0:1.0:20.0)),
    )

    table.sin[[2, 3, 7, 12, 19]] .= missing

    @test Filter() == Filter()

    @testset "Vector" begin
        result = apply(a, Filter())
        expected = deleteat!(deepcopy(a), [2, 3, 7])

        @test result == expected
        @test result == Impute.filter(a)

        a2 = deepcopy(a)
        Impute.filter!(a2)
        @test a2 == expected
    end

    @testset "Matrix" begin
        @test_throws UndefKeywordError apply(m, Filter())
        @test_throws UndefKeywordError Impute.filter(m)
        @test_throws MethodError Impute.filter!(m)

        @testset "rows" begin
            result = apply(m, Filter(); dims=:rows)
            expected = m[[1, 4, 5], :]

            @test isequal(result, expected)
            @test isequal(result, (Impute.filter(m; dims=:rows)))
            @test isequal(collect(result'), Impute.filter(collect(m'); dims=:cols))
        end

        @testset "cols" begin
            result = apply(m, Filter(); dims=:cols)
            expected = copy(m)[:, 3:4]

            @test isequal(result, expected)
            @test isequal(result, Impute.filter(m; dims=:cols))
            @test isequal(collect(result'), Impute.filter(collect(m'); dims=:rows))
        end
    end

    @testset "Tables" begin
        @testset "DataFrame" begin
            df = deepcopy(table)

            @test_throws UndefKeywordError apply(df, Filter())
            @test_throws UndefKeywordError Impute.filter(df)
            @test_throws MethodError Impute.filter!(df)

            @testset "rows" begin
                result = apply(df, Filter(); dims=:rows)
                expected = dropmissing(df)

                @test isequal(result, expected)
                @test isequal(result, Impute.filter(df; dims=:rows))
            end

            @testset "cols" begin
                result = apply(df, Filter(); dims=:cols)
                expected = select(df, :cos)

                @test isequal(result, expected)
                @test isequal(result, Impute.filter(df; dims=:cols))
            end
        end

        @testset "Column Table" begin
            coltab = Tables.columntable(table)

            @test_throws UndefKeywordError apply(coltab, Filter())
            @test_throws UndefKeywordError Impute.filter(coltab)
            @test_throws MethodError Impute.filter!(coltab)

            @testset "rows" begin
                result = apply(coltab, Filter(); dims=:rows)
                expected = Tables.columntable(dropmissing(table))

                @test isequal(result, expected)
                @test isequal(result, Impute.filter(coltab; dims=:rows))
            end

            @testset "cols" begin
                result = apply(coltab, Filter(); dims=:cols)
                expected = Tables.columntable(TableOperations.select(coltab, :cos))

                @test isequal(result, expected)
                @test isequal(result, Impute.filter(coltab; dims=:cols))
            end
        end

        @testset "Row Table" begin
            @testset "rows" begin
                rowtab = Tables.rowtable(table)

                result = apply(rowtab, Filter(); dims=:rows)
                expected = Tables.rowtable(dropmissing(table))

                @test isequal(result, expected)
                @test isequal(result, Impute.filter(rowtab; dims=:rows))
                @test isequal(result, Impute.filter(rowtab))

                rowtab_ = Impute.filter!(rowtab)
                @test isequal(rowtab, expected)
                @test isequal(rowtab_, expected)
            end

            @testset "cols" begin
                rowtab = Tables.rowtable(table)

                @test_throws ArgumentError Impute.filter!(rowtab; dims=:cols)
                result = apply(rowtab, Filter(); dims=:cols)
                expected = Tables.rowtable(TableOperations.select(rowtab, :cos))

                @test isequal(result, expected)
                @test isequal(result, Impute.filter(rowtab; dims=:cols))
            end

        end
    end

    @testset "AxisArray" begin
        @test_throws UndefKeywordError apply(aa, Filter())
        @test_throws UndefKeywordError Impute.filter(aa)
        @test_throws MethodError Impute.filter!(aa)

        @testset "rows" begin
            result = apply(aa, Filter(); dims=:rows)
            expected = m[[1, 4, 5], :]

            @test isequal(result, expected)
            @test isequal(result, Impute.filter(aa; dims=:rows))
            @test isequal(collect(result'), Impute.filter(collect(aa'); dims=:cols))
        end

        @testset "cols" begin
            result = apply(aa, Filter(); dims=:cols)
            expected = copy(aa)[:, 3:4]

            @test isequal(result, expected)
            @test isequal(result, Impute.filter(aa; dims=:cols))
            @test isequal(collect(result'), Impute.filter(collect(aa'); dims=:rows))
        end
    end
end
