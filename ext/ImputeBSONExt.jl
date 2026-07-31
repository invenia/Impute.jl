module ImputeBSONExt

using Impute
using BSON

Impute.load_bson(fullpath) = BSON.load(fullpath)

end  # module
