@testset "SVD" begin
    @testset "Default" begin
        tester = ImputorTester(SVD)
        test_hashing(tester)
        test_equality(tester)

        # test_matrix(tester)
        # Default transpose test uses `isequal`, but the SVD imputor will have a
        # small floating point error.
        @testset "Matrix" begin
            a = allowmissing(1.0:1.0:20.0)
            a[[2, 3, 7]] .= missing
            m = collect(reshape(a, 5, 4))

            result = impute(m, tester.imp(; tester.kwargs...); dims=:cols)

            @testset "Base" begin
                # Test that we have fewer missing values
                @test count(ismissing, result) < count(ismissing, m)
                @test isa(result, Matrix)
                @test eltype(result) <: eltype(m)

                # Test that functional form behaves the same way
                @test result == tester.f(m; dims=:cols, tester.kwargs...)
            end

            @testset "In-place" begin
                # Test that the in-place function return the new results and logs whether it
                # successfully did it in-place
                m2 = deepcopy(m)
                m2_ = tester.f!(m2; dims=:cols, tester.kwargs...)
                @test m2_ == result
                if m2 != result
                    @warn "$(tester.f!) did not mutate input data of type Matrix"
                end
            end

            @testset "Transpose" begin
                m_ = collect(m')
                result_ = collect(result')
                @test isapprox(tester.f(m_; dims=:rows, tester.kwargs...), result_)
                @test isapprox(tester.f!(m_; dims=:rows, tester.kwargs...), result_)
            end

            @testset "No missing" begin
                # Test having no missing data
                b = collect(reshape(allowmissing(1.0:1.0:20.0), 5, 4))
                @test impute(b, tester.imp(; tester.kwargs...); dims=:cols) == b
            end

            @testset "All missing" begin
                # Test having only missing data
                c = missings(5, 2)
                @test isequal(impute(c, tester.imp(; tester.kwargs...); dims=:cols), c)
            end
        end
        # Internal `svd` call isn't supported by these type, but maybe they should be?
        # test_axisarray(tester)
        # test_nameddimsarray(tester)
        # test_keyedarray(tester)
    end

    # Test a case where we expect SVD to perform well (e.g., many variables, )
    @testset "Data match" begin
        data = mapreduce(hcat, 1:1000) do i
            seeds = [sin(i), cos(i), tan(i), atan(i)]
            mapreduce(vcat, combinations(seeds)) do args
                [
                    +(args...),
                    *(args...),
                    +(args...) * 100,
                    +(abs.(args)...),
                    (+(args...) * 10) ^ 2,
                    (+(abs.(args)...) * 10) ^ 2,
                    log(+(abs.(args)...) * 100),
                    +(args...) * 100 + rand(-10:0.1:10),
                ]
            end
        end

        # println(svd(data').S)
        X = add_missings(data')

        svd_imputed = Impute.svd(X; dims=:cols)
        mean_imputed = Impute.fill(copy(X); dims=:cols)

        # With sufficient correlation between the variables and enough observation we
        # expect the svd imputation to perform severl times better than mean imputation.
        @test nrmsd(svd_imputed, data') < nrmsd(mean_imputed, data') * 0.5
    end

    # Test a case where we know SVD imputation won't perform well
    # (e.g., only a few variables, only )
    @testset "Data mismatch - too few variables" begin
        data = Tables.matrix(Impute.dataset("test/table/electricity"))
        X = add_missings(data)

        svd_imputed = Impute.svd(X; dims=:cols)
        mean_imputed = Impute.fill(copy(X); dims=:cols)

        # If we don't have enough variables then SVD imputation will probably perform
        # about as well as mean imputation.
        @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
    end

    @testset "Data mismatch - poor low rank approximations" begin
        M = rand(100, 200)
        data = M * M'
        X = add_missings(data)

        svd_imputed = Impute.svd(X; dims=:cols)
        mean_imputed = Impute.fill(copy(X); dims=:cols)

        # If most of the variance in the original data can't be explained by a small
        # subset of the eigen values in the svd decomposition then our low rank approximations
        # won't perform very well.
        @test nrmsd(svd_imputed, data) > nrmsd(mean_imputed, data) * 0.9
    end
end
