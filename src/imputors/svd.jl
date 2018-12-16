"""
    SVD <: Imputor

Imputes the missing values in a matrix using an expectation maximization (EM) algorithm
over low-rank SVD approximations.
"""
struct SVD <: Imputor
    init::Fill
    rank::Union{Int, Nothing}
    tol::Float64
    maxiter::Int
    limits::Union{Tuple{Float64, Float64}, Nothing}
    verbose::Bool
end

function SVD(; init=Fill(), rank=nothing, tol=1e-10, maxiter=10000, limits=nothing, verbose=true)
    SVD(init, rank, tol, maxiter, limits, verbose)
end

"""
    impute!(imp::SVD, ctx::Context, data::AbstractMatrix)


"""
function impute!(imp::SVD, ctx::Context, data::AbstractMatrix{<:Union{T, Missing}}) where T<:Real
    n, p = size(data)
    k = imp.rank === nothing ? 1 : min(imp.rank, p)
    S = zeros(T, p)
    X = zeros(T, n, p)

    # Get our before and after views of our missing and non-missing data
    mmask = ismissing.(Ref(ctx), data)
    omask = .!mmask

    mdata = data[mmask]
    mX = X[mmask]
    odata = data[omask]
    oX = X[omask]

    # Fill in the original data
    impute!(imp.init, ctx, data)

    C = sum((mdata - mX) .^ 2) / sum(mdata .^ 2)
    err = mean(abs.(odata - oX))
    @info("Before: Diff=$(sum(mdata - mX)), MAE=$err, convergence=$C, normsq=$(sum(mdata .^2)), $(mX[1])")

    for i in 1:imp.maxiter
        if imp.rank === nothing
            k = min(k + 1, p)
        end

        # Compute the SVD and produce a low-rank approximation of the data
        F = LinearAlgebra.svd(data)
        S[1:k] .= F.S[1:k]
        X = F.U * Diagonal(S) * F.Vt

        # Clamp the values if necessary
        imp.limits !== nothing && clamp!(X, imp.limits...)

        # Test for convergence
        mdata = data[mmask]
        mX = X[mmask]
        odata = data[omask]
        oX = X[omask]

        C = sum((mdata - mX) .^ 2) / sum(mdata .^ 2)

        # Print the error between reconstruction and observed inputs
        if imp.verbose
            err = mean(abs.(odata - oX))
            @info("Iteration $i: Diff=$(sum(mdata - mX)), MAE=$err, convergence=$C, normsq=$(sum(mdata .^2)), $(mX[1])")
        end

        # Update missing values
        data[mmask] .= X[mmask]

        if isfinite(C) && C < imp.tol
            break
        end
    end

    return data
end
