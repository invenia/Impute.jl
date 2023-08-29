function register_datadep()
    register(
        DataDep(
            "impute-v1.0.0",
            "Datasets for testing and demonstrating Impute.jl",
            "https://www.dropbox.com/scl/fi/bgtfqea9qqoug42gcnnsl/datasets.tar.gz?rlkey=11xsae0wi32m8gcfxhbqgo030&dl=0",
            "cf1fff2e7f3ce28eb4264060bc7b9ee561bccfce2c5915c4cf758ec48477ddfe",
            fetch_method=DataDeps.fetch_base,
            post_fetch_method=DataDeps.unpack,
        )
    )
end

function datasets()
    dep = datadep"impute-v1.0.0/data/"

    # Only select paths containing a data.x file
    selected = Iterators.filter(walkdir(dep)) do (root, dirs, files)
        any(f -> first(splitext(f)) == "data", files)
    end

    # Return just the root path with the data dep path part removed
    return [first(t)[length(dep)+2:end] for t in selected]
end

function dataset(name)
    dep = @datadep_str joinpath("impute-v1.0.0/data", name)
    files = readdir(dep)
    idx = findfirst(f -> first(splitext(f)) == "data", files)
    idx === nothing && throw(ArguementError("No data file found for $name."))
    fullpath = joinpath(dep, files[idx])
    ext = splitext(fullpath)[end]

    # This is necessary because CSV isn't registered in FileIO
    if ext == ".csv"
        return CSV.File(fullpath)
    elseif ext == ".bson"
        return BSON.load(fullpath)
    else
        throw(ArgumentError("Unsupported file type $ext."))
    end
end
