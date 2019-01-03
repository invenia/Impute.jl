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
    context::AbstractContext
end

function SVD(;
    init=Fill(), rank=nothing, tol=1e-10, maxiter=100, limits=nothing, verbose=true, context=Context()
)
    SVD(init, rank, tol, maxiter, limits, verbose, context)
end

"""
    impute!(imp::SVD, ctx::Context, data::AbstractMatrix)


"""
function impute!(data::AbstractMatrix{<:Union{T, Missing}}, imp::SVD) where T<:Real
    n, p = size(data)
    k = imp.rank === nothing ? 0 : min(imp.rank, p-1)
    S = zeros(T, min(n, p))
    X = zeros(T, n, p)

    ctx = imp.context
    # Get our before and after views of our missing and non-missing data
    mmask = ismissing.(data)
    omask = .!mmask

    mdata = data[mmask]
    mX = X[mmask]
    odata = data[omask]
    oX = X[omask]

    # Fill in the original data
    impute!(data, imp.init)

    C = sum(abs2, mdata - mX) / sum(abs2, mdata)
    err = mean(abs.(odata - oX))
    @info("Before: Diff=$(sum(mdata - mX)), MAE=$err, convergence=$C, normsq=$(sum(abs2, mdata)), $(mX[1])")

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
            @info("Iteration $i: Diff=$(sum(mdata - mX)), MAE=$err, MSS=$(sum(abs2, mdata)), convergence=$C")
        end

        # Update missing values
        data[mmask] .= X[mmask]

        if isfinite(C) && C < imp.tol
            break
        end
    end

    return data
end
