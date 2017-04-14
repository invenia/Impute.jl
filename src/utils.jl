"""
    convert{T<:Any}(::Type{T}, x::Nullable)

Converts the value of a Nullable to the specified type. Needed for casting
`Nullable`s to `Real`s for use in `UnitRange`s
"""
Base.convert{T<:Any}(::Type{T}, x::Nullable) = convert(T, x.value)

"""
    dataset(m::Module, package_name::AbstractString, Dataset_name::AbstractString)

Function for optional loading of test RDatasets into either DataFrames or DataTables
"""
function dataset(m::Module, package_name::AbstractString, Dataset_name::AbstractString)
    basename = joinpath(Pkg.dir("RDatasets"), "data", package_name)

    filename = joinpath(basename, string(Dataset_name, ".csv.gz"))
    if !isfile(filename)
        error(@sprintf "Unable to locate file %s or %s\n" rdaname filename)
    else
        return m.readtable(filename)
    end
end

"""
    filter!(f::Function, a::Union{NullableArray, DataArray})

Allows filtering on NullableArrays and DataArrays, this is pretty much copy-paste from
base julia, but they only supports `filter!` on `Array{T, 1}` since not all `AbstractVector`s
will implement `@deleteat!`.
"""
function Base.filter!(f::Function, a::Union{NullableArray, DataArray})
    insrt = 1
    for acurr in a
        if f(acurr)
            a[insrt] = acurr
            insrt += 1
        end
    end
    deleteat!(a, insrt:length(a))
    return a
end
