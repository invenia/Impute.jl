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

    # Test unsupported file type error (tests line 60 of ImputeDataDepsExt.jl)
    @testset "unsupported file type" begin
        # Add a test dataset with unsupported file type to the existing datasets
        dep = @datadep_str "impute-v1.0.0/data/"
        testdir = joinpath(dep, "test-unsupported")

        try
            # Create test directory with unsupported file extension
            mkpath(testdir)
            write(joinpath(testdir, "data.txt"), "test content")

            # Call Impute.dataset() which will execute line 60 and throw the error
            @test_throws ArgumentError("Unsupported file type .txt.") begin
                Impute.dataset("test-unsupported")
            end
        finally
            # Clean up the test directory
            rm(testdir; recursive=true, force=true)
        end
    end
end
