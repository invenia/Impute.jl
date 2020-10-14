@testset "data" begin
    datasets = Impute.datasets()

    @testset "Impute.dataset($name)" for name in datasets
        result = Impute.dataset(name)
        if occursin("matrix", name)
            @test isa(result, AbstractDict)
        elseif occursin("table", name)
            @test isa(result, CSV.File)
        end
    end
end
