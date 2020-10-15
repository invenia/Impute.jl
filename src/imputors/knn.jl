"""
    KNN(; kwargs...)

Imputation using k-Nearest Neighbor algorithm.

# Keyword Arguments
* `k::Int`: number of nearest neighbors
* `dist::MinkowskiMetric`: distance metric suppports by `NearestNeighbors.jl` (Euclidean, Chebyshev, Minkowski and Cityblock)
* `threshold::AbstractFloat`: threshold for missing neighbors

# Reference
* Troyanskaya, Olga, et al. "Missing value estimation methods for DNA microarrays." Bioinformatics 17.6 (2001): 520-525.
"""
struct KNN{M} <: Imputor where M <: NearestNeighbors.MinkowskiMetric
    k::Int
    threshold::AbstractFloat
    dist::M
end

# TODO : Support Categorical Distance (NearestNeighbors.jl support needed)
function KNN(; k=1, threshold=0.5, dist=Euclidean())
    k < 1 && throw(ArgumentError("The number of nearset neighbors should be greater than 0"))

    !(0 < threshold < 1) && throw(ArgumentError("Missing neighbors threshold should be within 0 to 1"))

    # to exclude missing value itself
    KNN(k + 1, threshold, dist)
end

function impute!(data::AbstractMatrix{Union{T, Missing}}, imp::KNN; dims=nothing) where T<:Real
    d = dim(data, dims)

    # KDTree expects data of the form dims x n
    X = d == 1 ? data : transpose(data)

    # Get mask array first
    Xmask = ismissing.(X)

    # Fill missing value as mean value
    impute!(X, Substitute(); dims=1)

    # Disallow `missings` for NearestNeighbors
    X = disallowmissing(X)

    # Our search points are just observations containing `missing`s
    points = X[:, vec(reduce(|, Xmask; dims=1))]

    # Contruct our KDTree over the entire dataset
    kdtree = KDTree(X, imp.dist)

    # Query for neighbors to our missing observations
    # NOTES:
    # 1. It's generally faster to query for all points at once
    # 2. We wanted the results sorted so that the first idx is our data points
    #   location in the original dataset.
    for (idx, dist) in zip(NearestNeighbors.knn(kdtree, points, imp.k, true)...)
        # Our closest neighbor should always be our input data point (distance of zero)
        @assert iszero(first(dist))

        # Location of point to impute
        j = first(idx)

        # Update each missing value in this point
        for i in 1:size(points, 1)
            # Skip non-missing elements
            Xmask[i, j] || continue

            # Grab our neighbor mask to excluding neighbor values that were also missing.
            nmask = Xmask[i, idx]

            # Skip if there are too many missing neighbor values
            (count(nmask) / imp.k) > imp.threshold && continue

            # Weight valid neighbors based on inverse distance
            wv = weights(1.0 ./ dist[.!nmask])

            # Only fill with the weighted mean of neighbors if the sum of the weights are
            # non-zero and finite.
            if isfinite(sum(wv)) && !iszero(sum(wv))
                X[i, j] = mean(X[i, idx[.!nmask]], wv)
            end
        end
    end

    # for type stability
    return allowmissing(d == 1 ? X : X')
end

impute!(data::AbstractMatrix{Missing}, imp::KNN; kwargs...) = data

function impute(data::AbstractMatrix{Union{T, Missing}}, imp::KNN; kwargs...) where T<:Real
    return impute!(trycopy(data), imp; kwargs...)
end

impute(data::AbstractMatrix{Missing}, imp::KNN; kwargs...) = trycopy(data)
