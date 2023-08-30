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
    missing_mask = ismissing.(X)

    # Fill missing value as mean value
    impute!(X, Substitute(); dims=1)

    # Disallow `missings` for NearestNeighbors
    X = disallowmissing(X)

    # Our search points are just observations containing `missing`s
    points = X[:, vec(any(missing_mask; dims=1))]

    # Contruct our KDTree over the entire dataset
    kdtree = KDTree(X, imp.dist)

    # Query for neighbors to our missing observations
    # NOTES:
    # 1. It's generally faster to query for all points at once
    # 2. We wanted the results sorted so that the first idx is our data points
    #   location in the original dataset.
    for (idxs, dists) in zip(NearestNeighbors.knn(kdtree, points, imp.k, true)...)
        # Our closest neighbor should always be our input data point (distance of zero)
        @assert iszero(first(dists))

        # Location of point to impute
        j = first(idxs)

        # Update each missing value in this point
        for i in axes(points, 1)
            # Skip non-missing elements
            missing_mask[i, j] || continue

            # Grab our neighbor mask to excluding neighbor values that were also missing.
            neighbor_mask = missing_mask[i, idxs]

            # Skip if there are too many missing neighbor values
            (count(neighbor_mask) / imp.k) > imp.threshold && continue

            # Weight valid neighbors based on inverse distance
            neighbor_dists = dists[.!neighbor_mask]

            # Delay creating Weights as they don't support Inf or NaN anymore.
            wv = 1.0 ./ neighbor_dists
            Σ = sum(wv)

            # Only fill with the weighted mean of neighbors if the sum of the weights are
            # non-zero and finite.
            if isfinite(Σ) && !iszero(Σ)
                neighbor_vals = X[i, idxs[.!neighbor_mask]]
                X[i, j] = mean(neighbor_vals, Weights(wv, Σ))
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
