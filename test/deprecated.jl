@testset "deprecated" begin
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

    @testset "Default colwise" begin
        msg = string(
            "Imputing on matrices will require specifying `dims=2` or `dims=:cols` in a ",
            "future release, to maintain the current behaviour."
        )
        @test_logs (:warn, msg) Impute.fill(m)
        @test_logs (:warn, msg) Impute.interp(m)
        @test_logs (:warn, msg) Impute.locf(m)
        @test_logs (:warn, msg) Impute.nocb(m)
        @test_logs (:warn, msg) Impute.srs(m)
    end

    @testset "Drop" begin
        @testset "Equality" begin
            @testset "$T" for T in (DropObs, DropVars)
                @test @test_deprecated(T()) == @test_deprecated(T())
            end
        end

        @testset "DropObs" begin
            @testset "Vector" begin
                result = @test_deprecated impute(a, DropObs())
                expected = deleteat!(deepcopy(a), [2, 3, 7])

                @test result == expected
                @test result == @test_deprecated Impute.dropobs(a)

                a2 = deepcopy(a)
                @test_deprecated Impute.dropobs!(a2)
                @test a2 == expected
            end

            @testset "Matrix" begin
                result = @test_deprecated impute(m, DropObs())
                expected = m[[1, 4, 5], :]

                @test isequal(result, expected)
                @test isequal(result, @test_deprecated(Impute.dropobs(m)))
                @test isequal(
                    collect(result'),
                    @test_deprecated(Impute.dropobs(collect(m'); dims=2))
                )

                m_ = @test_deprecated Impute.dropobs!(m)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(m, expected)
                @test isequal(m_, expected)
            end

            @testset "Tables" begin
                @testset "DataFrame" begin
                    df = deepcopy(table)
                    result = impute(df, @test_deprecated DropObs())
                    expected = dropmissing(df)

                    @test isequal(result, expected)
                    @test isequal(result, @test_deprecated(Impute.dropobs(df)))

                    df_ = @test_deprecated Impute.dropobs!(df)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(df, expected)
                    @test isequal(df_, expected)
                end

                @testset "Column Table" begin
                    coltab = Tables.columntable(table)

                    result = @test_deprecated impute(coltab, DropObs())
                    expected = Tables.columntable(dropmissing(table))

                    @test isequal(result, expected)
                    @test isequal(result, @test_deprecated(Impute.dropobs(coltab)))

                    coltab_ = @test_deprecated Impute.dropobs!(coltab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(coltab, expected)
                    @test isequal(coltab_, expected)
                end

                @testset "Row Table" begin
                    rowtab = Tables.rowtable(table)
                    result = @test_deprecated impute(rowtab, DropObs())
                    expected = Tables.rowtable(dropmissing(table))

                    @test isequal(result, expected)
                    @test isequal(result, @test_deprecated(Impute.dropobs(rowtab)))

                    rowtab_ = @test_deprecated Impute.dropobs!(rowtab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    # @test_broken isequal(rowtab, expected)
                    @test isequal(rowtab_, expected)
                end
            end

            @testset "AxisArray" begin
                result = @test_deprecated impute(aa, DropObs())
                expected = aa[[1, 4, 5], :]

                @test isequal(result, expected)
                @test isequal(result, @test_deprecated(Impute.dropobs(aa)))

                aa_ = @test_deprecated Impute.dropobs!(aa)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(aa, expected)
                @test isequal(aa_, expected)
            end
        end

        @testset "DropVars" begin
            @testset "Vector" begin
                @test_deprecated @test_throws MethodError Impute.dropvars(a)
            end

            @testset "Matrix" begin
                result = @test_deprecated impute(m, DropVars())
                expected = copy(m)[:, 3:4]

                @test isequal(result, expected)
                @test isequal(result, @test_deprecated(Impute.dropvars(m)))
                @test isequal(
                    collect(result'),
                    @test_deprecated(Impute.dropvars(collect(m'); dims=2))
                )

                m_ = @test_deprecated Impute.dropvars!(m)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(m, expected)
                @test isequal(m_, expected)
            end

            @testset "Tables" begin
                @testset "DataFrame" begin
                    df = deepcopy(table)
                    result = @test_deprecated impute(df, DropVars())
                    expected = select(df, :cos)

                    @test isequal(result, expected)
                    @test isequal(result, @test_deprecated Impute.dropvars(df))

                    @test_deprecated Impute.dropvars!(df)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(df, expected)
                end

                @testset "Column Table" begin
                    coltab = Tables.columntable(table)

                    result = @test_deprecated impute(coltab, DropVars())
                    expected = Tables.columntable(TableOperations.select(coltab, :cos))

                    @test isequal(result, expected)
                    @test isequal(result, @test_deprecated Impute.dropvars(coltab))

                    @test_deprecated Impute.dropvars!(coltab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(coltab, expected)
                end

                @testset "Row Table" begin
                    rowtab = Tables.rowtable(table)
                    result = @test_deprecated impute(rowtab, DropVars())
                    expected = Tables.rowtable(TableOperations.select(rowtab, :cos))

                    @test isequal(result, expected)
                    @test isequal(result, @test_deprecated Impute.dropvars(rowtab))

                    @test_deprecated Impute.dropvars!(rowtab)
                    # The mutating test is broken because we need to making a copy of
                    # the original table
                    @test_broken isequal(rowtab, expected)
                end
            end
            @testset "AxisArray" begin
                result = @test_deprecated impute(aa, DropVars())
                expected = copy(aa)[:, 3:4]

                @test isequal(result, expected)
                @test isequal(result, @test_deprecated Impute.dropvars(aa))

                aa_ = @test_deprecated Impute.dropvars!(aa)
                # The mutating test is broken because we need to making a copy of
                # the original matrix
                @test_broken isequal(aa, expected)
                @test isequal(aa_, expected)
            end
        end
    end

    @testset "utils" begin
        M = [1.0 2.0 3.0 4.0 5.0; 1.1 2.2 3.3 4.4 5.5]

        @testset "obswise" begin
            @test map(sum, Impute.obswise(M; dims=2)) == [2.1, 4.2, 6.3, 8.4, 10.5]
            @test map(sum, Impute.obswise(M; dims=1)) == [15, 16.5]
        end

        @testset "varwise" begin
            @test map(sum, Impute.varwise(M; dims=2)) == [15, 16.5]
            @test map(sum, Impute.varwise(M; dims=1)) == [2.1, 4.2, 6.3, 8.4, 10.5]
        end

        @testset "filterobs" begin
            @test Impute.filterobs(x -> sum(x) > 5.0, M; dims=2) == M[:, 3:5]
            @test Impute.filterobs(x -> sum(x) > 15.0, M; dims=1) == M[[false, true], :]
        end

        @testset "filtervars" begin
            @test Impute.filtervars(x -> sum(x) > 15.0, M; dims=2) == M[[false, true], :]
            @test Impute.filtervars(x -> sum(x) > 5.0, M; dims=1) == M[:, 3:5]
        end
    end
end
