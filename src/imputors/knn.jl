"""
    KNN <: Imputor

Imputation using k-Nearest Neighbor algorithm.

# Keyword Arguments
* `k::Int`: number of nearest neighbors
* `dist::MinkowskiMetric`: distance metric suppports by `NearestNeighbors.jl` (Euclidean, Chebyshev, Minkowski and Cityblock)
* `threshold::AbsstractFloat`: thershold for missing neighbors
* `on_complete::Function`: a function to run when imputation is complete

# Reference
* Troyanskaya, Olga, et al. "Missing value estimation methods for DNA microarrays." Bioinformatics 17.6 (2001): 520-525.
"""
# TODO : Support Categorical Distance (NearestNeighbors.jl support needed)
struct KNN{M} <: Imputor where M <: NearestNeighbors.MinkowskiMetric
    k::Int
    threshold::AbstractFloat
    dist::M
end

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
    mmask = ismissing.(X)

    # fill missing value as mean value
    impute!(X, Substitute(); dims=1)

    # Disallow `missings` for NearestNeighbors
    X = disallowmissing(X)

    kdtree = KDTree(X, imp.dist)
    idxs, dists = NearestNeighbors.knn(kdtree, X, imp.k, true)

    idxes = CartesianIndices(X)
    fallback_threshold = imp.k * imp.threshold

    for I in CartesianIndices(X)
        if mmask[I] == 1
            w = 1.0 ./ dists[I[2]]
            ws = sum(w[2:end])
            # Shouldn't ismissing.(X[...][...]) be replaced with mmask[...][...]?
            # If so then I think the test might need updating cause the "Data match" section
            # seems to fallback on the mean imputation consistently
            neighbors = mapslices(
                iszero ∘ sum,
                ismissing.(X[:, idxs[I[2]]][:, 2:end]);
                dims=1
            )

            # exclude missing value itself because distance would be zero
            # If too many neighbors are also missing, fallback to mean imputation
            # get column and check how many neighbors are also missing
            if isfinite(ws) && !iszero(ws) && count(neighbors) > fallback_threshold
                # Inverse distance weighting
                wt = w .* X[I[1], idxs[I[2]]]
                X[I] = sum(wt[2:end]) / ws
            end
        end
    end

    # for type stability
    return allowmissing(d == 1 ? X : X')
end

function impute(data::AbstractMatrix{Union{T, Missing}}, imp::KNN) where T<:Real
    return impute!(trycopy(data), imp)
end
