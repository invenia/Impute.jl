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

function impute!(data::AbstractMatrix{<:Union{T, Missing}}, imp::KNN) where T<:Real
    # Get mask array first (order of )
    mmask = ismissing.(transpose(data))

    # fill missing value as mean value
    impute!(data, Fill(; value=mean))

    # then, transpose to D x N for KDTree
    transposed = transpose(disallowmissing(data))

    kdtree = KDTree(transposed, imp.dist)
    idxs, dists = NearestNeighbors.knn(kdtree, transposed, imp.k, true)

    idxes = CartesianIndices(transposed)
    fallback_threshold = imp.k * imp.threshold

    for I in CartesianIndices(transposed)
        if mmask[I] == 1
            w = 1.0 ./ dists[I[2]]
            ws = sum(w[2:end])
            missing_neighbors = ismissing.(transposed[:, idxs[I[2]]][:, 2:end])

            # exclude missing value itself because distance would be zero
            if isnan(ws) || isinf(ws) || iszero(ws)
                # if distance is zero or not a number, keep mean imputation
                transposed[I] = transposed[I]
            elseif count(!iszero, mapslices(sum, missing_neighbors, dims=1)) >
                fallback_threshold
                # If too many neighbors are also missing, fallback to mean imputation
                # get column and check how many neighbors are also missing
                transposed[I] = transposed[I]
            else
                # Inverse distance weighting
                wt = w .* transposed[I[1], idxs[I[2]]]
                transposed[I] = sum(wt[2:end]) / ws
            end
        end
    end

    # for type stability
    allowmissing(transposed')
end
