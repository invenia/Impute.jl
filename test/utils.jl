@testset "Utilities" begin
    @testset "Impute.dim" begin
        X = rand(10, 5)
        KA = KeyedArray(X; A=1:10, B=collect("abcde"))

        @test Impute.dim(X, 1) == Impute.dim(X, :rows) == Impute.dim(KA, :A)
        @test first(eachslice(X, dims=1)) == first(eachslice(KA, dims=1)) == first(eachslice(KA, dims=:A))

        @test Impute.dim(X, 2) == Impute.dim(X, :cols) == Impute.dim(KA, :B)
        @test first(eachslice(X, dims=2)) == first(eachslice(KA, dims=2)) == first(eachslice(KA, dims=:B))
    end
end
