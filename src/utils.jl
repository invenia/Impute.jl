# A couple utility methods to avoid messing up var and obs dimensions
obsdim(dims::Int) = dims
vardim(dims::Int) = dims == 1 ? 2 : 1
obsdim(dims::Symbol) = dims === :row ? 1 : 2
vardim(dims::Symbol) = dims === :col ? 2 : 1

function obswise(data::AbstractMatrix; dims=1)
    return (selectdim(data, obsdim(dims), i) for i in axes(data, obsdim(dims)))
end

function varwise(data::AbstractMatrix; dims=2)
    return (selectdim(data, vardim(dims), i) for i in axes(data, vardim(dims)))
end

function filterobs(f::Function, data::AbstractMatrix; dims=1)
    mask = [f(x) for x in obswise(data; dims=dims)]
    return dims == 1 ? data[mask, :] : data[:, mask]
end

function filtervars(f::Function, data::AbstractMatrix; dims=2)
    mask = [f(x) for x in varwise(data; dims=dims)]
    return dims == 1 ? data[:, mask] : data[mask, :]
end
