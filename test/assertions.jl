@testset "Assertions" begin
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
        t = Threshold(; ratio=0.1)
        @test_throws AssertionError assert(a, t)
        @test_throws AssertionError assert(m, t)
        @test_throws AssertionError assert(aa, t)
        @test_throws AssertionError assert(table, t)

        t = Threshold(; ratio=0.8)
        # Use isequal because we expect the results to contain missings
        @test isequal(assert(a, t), a)
        @test isequal(assert(m, t), m)
        @test isequal(assert(aa, t), aa)
        @test isequal(assert(table, t), table)
    end

    @testset "Weighted" begin
        # If we use an exponentially weighted context then we won't pass the limit
        # because missing earlier observations is less important than later ones.
        t = Threshold(; ratio=0.8, weights=eweights(20, 0.3))
        @test isequal(assert(a, t), a)
        @test isequal(assert(table, t), table)

        @test isequal(threshold(m; ratio=0.8, weights=eweights(5, 0.3), dims=:cols), m)
        @test isequal(threshold(m; ratio=0.8, weights=eweights(5, 0.3), dims=:cols), aa)

        # If we reverse the weights such that earlier observations are more important
        # then our previous limit of 0.2 won't be enough to succeed.
        t = Threshold(; ratio=0.1, weights=reverse!(eweights(20, 0.3)))
        @test_throws AssertionError assert(a, t)
        @test_throws AssertionError assert(table, t)

        t = Threshold(; ratio=0.1, weights=reverse!(eweights(5, 0.3)))
        @test_throws AssertionError assert(m, t; dims=:cols)
        @test_throws AssertionError assert(aa, t; dims=:cols)

        @test_throws DimensionMismatch assert(a[1:10], t)
        @test_throws DimensionMismatch assert(m[1:3, :], t; dims=:cols)
    end

    @testset "functional" begin
        @test_throws AssertionError Impute.threshold(a; ratio=0.1)
        # Use isequal because we expect the results to contain missings
        @test isequal(Impute.threshold(a; ratio=0.8), a)
    end
end
