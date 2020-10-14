@testset "Substitute" begin
    @testset "Default" begin
        test_all(ImputorTester(Substitute))
    end

    @testset "defaultstats" begin
        @testset "robust" begin
            # Defining our missing datasets
            a = allowmissing(1.0:1.0:20.0)
            a[[2, 3, 7]] .= missing
            fill_val = median(skipmissing(a))

            result = impute(a, Substitute())
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.substitute(a)
        end

        @testset "weighted" begin
            # Defining our missing datasets
            a = allowmissing(1.0:1.0:20.0)
            wv = eweights(20, 0.3)
            a[[2, 3, 7]] .= missing
            mask = .!ismissing.(a)

            fill_val = mean(a[mask], wv[mask])

            result = impute(a, Substitute(; robust=false, weights=wv))
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.substitute(a; robust=false, weights=wv)
        end

        @testset "counts" begin
            a = allowmissing([1, 12, 4, 6, 2, 5, 9, 19, 24, 35, 44, 99])
            a[[2, 3, 7]] .= missing

            # We should default to taking the  median because otherwise `mode` will
            # just return `1`
            fill_val = round(Int, median(skipmissing(a)))

            result = impute(a, Substitute())
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.substitute(a)
        end

        @testset "ratings" begin
            # Slightly imbalanced ratings
            a = allowmissing(vcat(repeat(1:5, 5), [1, 1, 5]))
            a[[2, 3, 7]] .= missing

            # We likely want to the mode because we only have a few unique values.
            fill_val = mode(skipmissing(a))
            @test fill_val == 1

            result = impute(a, Substitute())
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.substitute(a)
        end

        @testset "bools" begin
            a = allowmissing(vcat(falses(14), trues(6)))
            a[[2, 3, 7]] .= missing

            # For the same reason as for ratings we should probably just use the mode.
            # Though most of the time they'll give the same answer once rounded.
            fill_val = mode(skipmissing(a))
            @test fill_val == false

            result = impute(a, Substitute())
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.substitute(a)
        end

        @testset "non-real" begin
            a = allowmissing(DateTime(2000, 1, 1):Day(1):DateTime(2000, 1, 20))
            a[[2, 3, 7]] .= missing

            # Median of `DateTime`s doesn't apply, so we fallback to `mode`
            fill_val = mode(skipmissing(a))

            # In this case that's just going to take the first observation it finds
            fill_val == DateTime(2000, 1, 1)

            result = impute(a, Substitute())
            expected = copy(a)
            expected[[2, 3, 7]] .= fill_val

            @test result == expected
            @test result == Impute.substitute(a)
        end
    end

    @testset "custom statistic" begin
        # Defining our missing datasets
        a = allowmissing(1.0:1.0:20.0)
        a[[2, 3, 7]] .= missing

        # We'll do mean - 1 std for some reason :)
        μ, σ = mean_and_std(skipmissing(a))
        fill_val = μ - σ

        expected = copy(a)
        expected[[2, 3, 7]] .= fill_val
        result = Impute.substitute(
            a;
            statistic=data -> -(mean_and_std(skipmissing(data))...)
        )
        @test result == expected
    end
end
