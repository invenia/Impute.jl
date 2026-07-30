module ImputeBSONExt

using Impute
using BSON

function Impute.load_bson(fullpath::AbstractString)
    return BSON.load(fullpath)
end

end  # module
