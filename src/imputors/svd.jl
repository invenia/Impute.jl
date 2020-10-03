"""
    SVD <: Imputor

Imputes the missing values in a matrix using an expectation maximization (EM) algorithm
over low-rank SVD approximations.

# Keyword Arguments
* `init::Imputor`: initialization method for missing values (default: Substitute())
* `rank::Union{Int, Nothing}`: rank of the SVD approximation (default: nothing meaning start and 0 and increase)
* `tol::Float64`: convergence tolerance (default: 1e-10)
* `maxiter::Int`: Maximum number of iterations if convergence is not achieved (default: 100)
* `limits::Unoin{Tuple{Float64, Float64}, Nothing}`: Bound the possible approximation values (default: nothing)
* `verbose::Bool`: Whether to display convergence progress (default: true)

# References
* Troyanskaya, Olga, et al. "Missing value estimation methods for DNA microarrays." Bioinformatics 17.6 (2001): 520-525.
"""
struct SVD <: Imputor
    init::Imputor
    rank::Union{Int, Nothing}
    tol::Float64
    maxiter::Int
    limits::Union{Tuple{Float64, Float64}, Nothing}
    verbose::Bool
end

function SVD(;
    init=Substitute(), rank=nothing, tol=1e-10, maxiter=100, limits=nothing, verbose=true
)
    SVD(init, rank, tol, maxiter, limits, verbose)
end

function impute!(data::AbstractMatrix{Union{T, Missing}}, imp::SVD; dims=nothing) where T<:Real
    n, p = size(data)
    k = imp.rank === nothing ? 0 : min(imp.rank, p-1)
    S = zeros(T, min(n, p))
    X = zeros(T, n, p)

    # Get our before and after views of our missing and non-missing data
    mmask = ismissing.(data)
    omask = .!mmask

    mdata = data[mmask]
    mX = X[mmask]
    odata = data[omask]
    oX = X[omask]

    # Fill in the original data
    impute!(data, imp.init; dims=dims)

    C = sum(abs2, mdata - mX) / sum(abs2, mdata)
    err = mean(abs.(odata - oX))
    @debug("Before: Diff=$(sum(mdata - mX)), MAE=$err, convergence=$C, normsq=$(sum(abs2, mdata)), $(mX[1])")

    for i in 1:imp.maxiter
        if imp.rank === nothing
            k = min(k + 1, p - 1, n - 1)
        end

        # Compute the SVD and produce a low-rank approximation of the data
        F = LinearAlgebra.svd(data)
        # println(join([size(S), size(F.S), size(F.U), size(F.Vt)], ", "))

        S[1:k] .= F.S[1:k]
        X = F.U * Diagonal(S) * F.Vt

        # Clamp the values if necessary
        imp.limits !== nothing && clamp!(X, imp.limits...)

        # Test for convergence
        mdata = data[mmask]
        mX = X[mmask]
        odata = data[omask]
        oX = X[omask]

        # println(join([size(mdata), size(mX)], ", "))
        C = sum(abs2, mdata - mX) / sum(abs2, mdata)

        # Print the error between reconstruction and observed inputs
        if imp.verbose
            err = mean(abs.(odata - oX))
            @debug("Iteration $i: Diff=$(sum(mdata - mX)), MAE=$err, MSS=$(sum(abs2, mdata)), convergence=$C")
        end

        # Update missing values
        data[mmask] .= X[mmask]

        if isfinite(C) && C < imp.tol
            break
        end
    end

    return data
end

function impute(data::AbstractMatrix{Union{T, Missing}}, imp::SVD) where T<:Real
    return impute!(trycopy(data), imp)
end
