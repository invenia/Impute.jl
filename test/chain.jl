@testset "Chaining and Piping" begin
    # TODO: Add tests at each section to double check that orig hasn't been overwritten.
    orig = Impute.dataset("test/table/neuro") |> DataFrame

    @testset "DataFrame" begin
        # Less effecient, but a chain should produce the same results as manual
        # piping the functional outputs.
        result = Impute.interp(orig) |> Impute.locf! |> Impute.nocb!

        @test size(result) == size(orig)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, Matrix(result))

        # We can also use the Chain type with explicit Imputor types
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result2 = C(orig)
        @test result == result2

        @testset "GroupedDataFrame" begin
            T = NamedTuple{(:hod, :obj, :val), Tuple{Int, Int, Union{Float64, Missing}}}

            df = map(Iterators.product(1:24, 1:8, 0:19)) do t
                hod, obj, x = t
                # Deterministically return some `missing`s per hod/obj pair
                return if x in (0, 5, 12, 19)
                    T((hod, obj, missing))
                else
                    T((hod, obj, sin(hod) * cos(x) + obj))
                end
            end |> DataFrame

            gdf1 = groupby(deepcopy(df), [:hod, :obj])
            gdf2 = groupby(df, [:hod, :obj])

            C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()

            result = mapreduce(C, vcat, gdf1)
            # Check that the result isn't the same as the source dataframe
            @test df != result
            # Check that the size is still the same since we didn't drop any rows
            @test size(result) == size(df)
            # Check that there are no remaining missing values
            @test all(!ismissing, Tables.matrix(result))
            # Double check that our source dataframe still contains missings
            @test any(ismissing, Tables.matrix(df))
        end
    end

    @testset "Column Table" begin
        result = Tables.columntable(orig) |>
            Impute.interp! |>
            Impute.locf! |>
            Impute.nocb! |>
            Tables.matrix

        @test size(result) == size(orig)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
    end

    @testset "Row Table" begin
        result = Tables.rowtable(orig) |>
            Impute.interp! |>
            Impute.locf! |>
            Impute.nocb! |>
            Tables.matrix

        @test size(result) == size(orig)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
    end

    @testset "Matrix" begin
        data = Matrix(orig)
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result = C(data; dims=:cols)

        @test size(result) == size(data)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
    end

    @testset "AxisArray" begin
        data = AxisArray(
            Matrix(orig),
            Axis{:row}(1:size(orig, 1)),
            Axis{:V}(names(orig)),
        )
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result = C(data; dims=:cols)

        @test size(result) == size(data)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
    end

    @testset "KeyedArray" begin
        data = KeyedArray(Matrix(orig); row=1:size(orig, 1), V=names(orig))
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result = C(data; dims=:cols)

        @test size(result) == size(data)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
    end

    @testset "Multi-type" begin
        data = Impute.dataset("test/table/neuro") |> Tables.matrix
        @test any(ismissing, data)
        # Filter out colunns with more than 400 missing values, Fill with 0, and check that
        # everything was replaced
        C = Chain(
            Impute.Filter(c -> count(ismissing, c) < 400),
            Impute.Replace(; values=0.0),
            Impute.Threshold(),
        )

        result = C(data; dims=:cols)
        @test size(result, 1) == size(data, 1)
        # We should have filtered out 1 column
        @test size(result, 2) < size(data, 2)
        @test all(!ismissing, result)
    end
end
