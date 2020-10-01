"""
    Filter([f])

Uses a function `f` to identify values, rows, columns or slices of data that should be
removed during an `apply` call. The default function `f` will removing `missing`s, or any
rows, columns or slices containing `missing`s.
"""
struct Filter
    func::Function
end

Filter() = Filter(_keep)

_keep(x) = !ismissing(x)
_keep(x::Union{Tuple, AbstractArray, IterTools.PropertyValues}) = all(_keep, x)

apply!(data::Vector, f::Filter) = Base.filter!(f.func, data)
function apply!(data::Vector{<:NamedTuple}, f::Filter; dims=:rows)
    d = dim(data, dims)
    d == 1 || throw(ArgumentError("Rowtables only support in-place filtering rowwise."))
    return Base.filter!(r -> f.func(propertyvalues(r)), data)
end

apply(data::Vector, f::Filter) = Base.filter(f.func, data)
function apply(data::Vector{<:NamedTuple}, f::Filter; dims=:rows)
    d = dim(data, dims)
    if d == 1
        return Base.filter(r -> f.func(propertyvalues(r)), data)
    else
        return materializer(data)(apply(Tables.columns(data), f; dims=dims))
    end
end

function apply(data::AbstractArray{Union{T, Missing}}, f::Filter; dims) where T
    d = dim(data, dims)
    mask = map(f.func, eachslice(data; dims=d))
    idx = (i == d ? mask : Colon() for i in 1:ndims(data))
    return data[idx...]
end

apply(data::AbstractArray, f::Filter) = disallowmissing(data)

function apply(table, f::Filter; dims)
    istable(table) || throw(MethodError(apply, (table, f)))

    d = dim(table, dims)
    filtered = if d == 1
        Iterators.filter(Tables.rows(table)) do r
            f.func(propertyvalues(r))
        end
    else
        cols = Tables.columns(table)

        cnames = Iterators.filter(propertynames(cols)) do cname
            f.func(getproperty(cols, cname))
        end

        TableOperations.select(table, cnames...)
    end

    return materializer(table)(filtered)
end
