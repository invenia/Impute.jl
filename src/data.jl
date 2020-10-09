function register_datadep()
    register(
        DataDep(
            "impute-v0.1.0",
            "Datasets for testing and demonstrating Impute.jl",
            "https://invenia-public-datasets.s3.amazonaws.com/Impute/v0.1.0/datasets.tar.gz",
            "938b3705752eb73141476a2abc7a36cfdaba9ec45f99f0796f44e0870e006e1c",
            post_fetch_method=unpack,
        )
    )
end

function datasets()
    dep = datadep"impute-v0.1.0/data/"

    # Only select paths containing a data.x file
    selected = Iterators.filter(walkdir(dep)) do (root, dirs, files)
        any(f -> first(splitext(f)) == "data", files)
    end

    # Return just the root path with the data dep path part removed
    return [first(t)[length(dep)+2:end] for t in selected]
end

function dataset(name)
    dep = @datadep_str joinpath("impute-v0.1.0/data", name)
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
