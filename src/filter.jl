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
_keep(t::Tuple) = all(_keep, t)
_keep(a::AbstractArray{Union{T, Missing}}) where {T} = all(_keep, a)
_keep(pv::IterTools.PropertyValues) = all(_keep, pv)

apply!(data::Vector, f::Filter) = Base.filter!(f.func, data)
function apply!(data::Vector{<:NamedTuple}, f::Filter; dims=:rows)
    if dims == 1 || dims == :rows
        return Base.filter!(r -> f.func(propertyvalues(r)), data)
    else
        throw(ArgumentError(
            "In-place filtering of rowtables is only guaranteed for row filtering."
        ))
    end
end

apply(data::Vector, f::Filter) = Base.filter(f.func, data)
function apply(data::Vector{<:NamedTuple}, f::Filter; dims=:rows)
    if dims == 1 || dims == :rows
        return Base.filter(r -> f.func(propertyvalues(r)), data)
    elseif dims == 2 || dims == :cols
        return materializer(data)(apply(Tables.columns(data), f; dims=dims))
    else
        throw(ArgumentError("Unknown `dims` value of $dims"))
    end
end

function apply(data::AbstractArray{Union{T, Missing}}, f::Filter; dims) where T
    # Support :rows and :cols for matrices
    if isa(data, AbstractMatrix)
        if dims == :rows
            dims = 1
        elseif dims == :cols
            dims = 2
        end
    end

    mask = map(f.func, eachslice(data; dims=dims))
    # NOTE: We're currently assuming dims is an integer at this point, but we should
    # also be able to support NamedDims.jl here once hasnames is provided.
    idx = (i == dims ? mask : Colon() for i in 1:ndims(data))
    return data[idx...]
end

apply(data::AbstractArray, f::Filter) = disallowmissing(data)

function apply(table, f::Filter; dims)
    istable(table) || throw(MethodError(apply, (table, f)))

    filtered = if dims == 1 || dims == :rows
        rows = Tables.rows(table)

        Iterators.filter(rows) do r
            x = propertyvalues(r)
            # @show x
            return f.func(x)
        end
    elseif dims == 2 || dims == :cols
        cols = Tables.columns(table)

        cnames = Iterators.filter(propertynames(cols)) do cname
            f.func(getproperty(cols, cname))
        end

        TableOperations.select(table, cnames...)
    else
        throw(ArgumentError("Unknown `dims` value of $dims"))
    end

    return materializer(table)(filtered)
end
