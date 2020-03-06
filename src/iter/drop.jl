function drop(xs)
    Iterators.filter(xs) do x
        return !(ismissing(x) || any(ismissing, x))
    end
end

function drop(xs, τ::Float64)
    Iterators.filter(xs) do x
        return (count(ismissing, x) / length(x)) < τ
    end
end


# TODO: Integrate with the threshold iterator?
