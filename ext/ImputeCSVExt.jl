module ImputeCSVExt

using Impute
using CSV

function Impute.load_csv(fullpath::AbstractString)
    return CSV.File(fullpath)
end

end  # module
