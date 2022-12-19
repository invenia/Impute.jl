@testset "Validators" begin
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

    @testset "Base" begin
        t = Threshold(; limit=0.1)
        @test_throws ThresholdError validate(a, t)
        @test_throws ThresholdError validate(m, t)
        @test_throws ThresholdError validate(aa, t)
        @test_throws ThresholdError validate(table, t)

        # Test showerror
        msg = try
            validate(a, t)
        catch e
            sprint(showerror, e)
        end

        @test msg == "ThresholdError: Missing data limit exceeded 0.1 (0.15)\n"

        t = Threshold(; limit=0.8)
        # Use isequal because we expect the results to contain missings
        @test isequal(validate(a, t), a)
        @test isequal(validate(m, t), m)
        @test isequal(validate(aa, t), aa)
        @test isequal(validate(table, t), table)

        # Test type edge cases
        @test_throws ThresholdError validate(fill(missing, 10), t)
        @test validate(ones(10), t) == ones(10)
        @test validate(ones(10), t) isa Vector{Float64}
    end

    @testset "Weighted" begin
        # If we use an exponentially weighted context then we won't pass the limit
        # because missing earlier observations is less important than later ones.
        t = WeightedThreshold(; limit=0.8, weights=eweights(20, 0.3))
        @test isequal(validate(a, t), a)
        @test isequal(validate(table, t), table)

        @test isequal(wthreshold(m; limit=0.8, weights=eweights(5, 0.3), dims=:cols), m)
        @test isequal(wthreshold(m; limit=0.8, weights=eweights(5, 0.3), dims=:cols), aa)

        # If we reverse the weights such that earlier observations are more important
        # then our previous limit of 0.2 won't be enough to succeed.
        t = WeightedThreshold(; limit=0.1, weights=reverse!(eweights(20, 0.3)))
        @test_throws ThresholdError validate(a, t)
        @test_throws ThresholdError validate(table, t)

        t = WeightedThreshold(; limit=0.1, weights=reverse!(eweights(5, 0.3)))
        @test_throws ThresholdError validate(m, t; dims=:cols)
        @test_throws ThresholdError validate(aa, t; dims=:cols)

        @test_throws DimensionMismatch validate(a[1:10], t)
        @test_throws DimensionMismatch validate(m[1:3, :], t; dims=:cols)

        @test_throws ThresholdError validate(fill(missing, 5), t)
        @test validate(ones(5), t) == ones(5)
        @test validate(ones(5), t) isa Vector{Float64}
    end

    @testset "functional" begin
        @test_throws ThresholdError Impute.threshold(a; limit=0.1)
        # Use isequal because we expect the results to contain missings
        @test isequal(Impute.threshold(a; limit=0.8), a)
    end
end
