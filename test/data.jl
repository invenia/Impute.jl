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

    # Test fetch_without_logs function directly
    @testset "fetch_without_logs" begin
        mktempdir() do tmpdir
            # Create a simple file to download
            test_file = joinpath(tmpdir, "source.txt")
            write(test_file, "test content")

            # Test that fetch_without_logs works
            dest_dir = joinpath(tmpdir, "dest")
            mkpath(dest_dir)

            # Call fetch_without_logs - it should copy/fetch the file
            result = Impute.fetch_without_logs("file://" * test_file, dest_dir)

            # Verify the file was fetched
            @test isfile(joinpath(dest_dir, "source.txt"))
        end
    end
end
