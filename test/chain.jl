@testset "Chaining and Piping" begin
    # `orig` should never be imputed as it is used as a reference for checking that the
    # `data` isn't being mutated below
    orig = Impute.dataset("test/table/neuro") |> DataFrame
    data = deepcopy(orig)

    @testset "DataFrame" begin
        # Less effecient, but a chain should produce the same results as manual
        # piping the functional outputs.
        result = Impute.interp(data) |> Impute.locf! |> Impute.nocb!

        @test size(result) == size(orig)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, Matrix(result))
        # Test we haven't mutated the data
        @test isequal(orig, data)

        # We can also use the Chain type with explicit Imputor types
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result2 = C(data)
        @test result == result2
        # Test we haven't mutated the data
        @test isequal(orig, data)

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
        result = Tables.columntable(data) |>
            Impute.interp |>
            Impute.locf! |>
            Impute.nocb! |>
            Tables.matrix

        @test size(result) == size(orig)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
        # Test we haven't mutated the data
        @test isequal(orig, data)
    end

    @testset "Row Table" begin
        result = Tables.rowtable(data) |>
            Impute.interp |>
            Impute.locf! |>
            Impute.nocb! |>
            Tables.matrix

        @test size(result) == size(orig)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
        # Test we haven't mutated the data
        @test isequal(orig, data)
    end

    @testset "Matrix" begin
        input = Matrix(data)
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result = C(input; dims=:cols)

        @test size(result) == size(input)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
        # Test we haven't mutated the data
        @test isequal(orig, data)
    end

    @testset "AxisArray" begin
        input = AxisArray(
            Matrix(orig),
            Axis{:row}(1:size(orig, 1)),
            Axis{:V}(names(orig)),
        )
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result = C(input; dims=:cols)

        @test size(result) == size(input)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
        # Test we haven't mutated the data
        @test isequal(orig, data)
    end

    @testset "KeyedArray" begin
        input = KeyedArray(Matrix(orig); row=1:size(orig, 1), V=names(orig))
        C = Impute.Interpolate() ∘ Impute.LOCF() ∘ Impute.NOCB()
        result = C(input; dims=:cols)

        @test size(result) == size(input)
        # Confirm that we don't have any more missing values
        @test all(!ismissing, result)
        # Test we haven't mutated the data
        @test isequal(orig, data)
    end

    @testset "Multi-type" begin
        input = Tables.matrix(data)
        @test any(ismissing, input)
        # Filter out colunns with more than 400 missing values, Fill with 0, and check that
        # everything was replaced
        C = Chain(
            Impute.Filter(c -> count(ismissing, c) < 400),
            Impute.Replace(; values=0.0),
            Impute.Threshold(),
        )

        result = C(input; dims=:cols)
        @test size(result, 1) == size(input, 1)
        # We should have filtered out 1 column
        @test size(result, 2) < size(input, 2)
        @test all(!ismissing, result)
        # Test we haven't mutated the data
        @test isequal(orig, data)
    end
end
