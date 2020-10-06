@testset "Replace" begin
    @testset "Default" begin
        # Tester that replaces with 0.0
        tester = ImputorTester(Replace; values=0.0)

        # Defining our own equality because an empty constructor isn't supported
        @testset "Equality" begin
            @test tester.imp(; tester.kwargs...) == tester.imp(; tester.kwargs...)
        end

        test_vector(tester)
        test_matrix(tester)
        test_dataframe(tester)
        # groupby test also fail because it tries to call an empty constructor
        # test_groupby(tester)
        test_axisarray(tester)
        test_nameddimsarray(tester)
        test_keyedarray(tester)
        test_columntable(tester)
        test_rowtable(tester)
    end

    @testset "Multiple values over tables" begin
        imp = Replace(; values=(DateTime(0), -9999, NaN, ""))
        df_table = DataFrame(
            :time => [missing, DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
            :loc => [12, -5, missing],
            :val => [1.5, missing, 3.0],
            :desc => ["foo", "bar", missing],
        )
        df_expected = DataFrame(
            :time => [DateTime(0), DateTime(2020, 02, 02), DateTime(2121, 12, 12)],
            :loc => [12, -5, -9999],
            :val => [1.5, NaN, 3.0],
            :desc => ["foo", "bar", ""],
        )

        @testset "DataFrame" begin
            table = copy(df_table)
            expected = copy(df_expected)

            result = impute(table, imp)
            @test isequal(result, expected)

            table2 = deepcopy(table)
            impute!(table2, imp)
            @test isequal(table2, expected)
        end

        @testset "Column Table" begin
            table = Tables.columntable(df_table)
            expected = Tables.columntable(df_expected)

            result = impute(table, imp)
            @test isequal(result, expected)

            table2 = deepcopy(table)
            impute!(table2, imp)
            @test isequal(table2, expected)
        end

        @testset "Row Table" begin
            table = Tables.rowtable(df_table)
            expected = Tables.rowtable(df_expected)

            result = impute(table, imp)
            @test isequal(result, expected)

            table2 = deepcopy(table)
            impute!(table2, imp)
            @test !isequal(table2, expected)
            @test isequal(table2, table)
        end
    end
end
