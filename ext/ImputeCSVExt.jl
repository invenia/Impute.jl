module ImputeCSVExt

using Impute
using CSV

Impute.load_csv(fullpath) = CSV.File(fullpath)

end  # module
