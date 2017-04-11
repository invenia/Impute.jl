
# Function option for loading test RDatasets into either DataFrames or DataTables
function dataset(m::Module, package_name::AbstractString, dataset_name::AbstractString)
    basename = joinpath(Pkg.dir("RDatasets"), "data", package_name)

    filename = joinpath(basename, string(dataset_name, ".csv.gz"))
    if !isfile(filename)
        error(@sprintf "Unable to locate file %s or %s\n" rdaname filename)
    else
        return m.readtable(filename)
    end
end
