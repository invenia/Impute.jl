@testset "KNN" begin
    @testset "Default" begin
        tester = ImputorTester(KNN)
        test_hashing(tester)
        test_equality(tester)
        test_matrix(tester)
        test_axisarray(tester)
        test_nameddimsarray(tester)
        test_keyedarray(tester)
    end
    @testset "Iris" begin
        # Reference
        # P. Schimitt, et. al
        # A comparison of six methods for missing data imputation
        iris = Impute.dataset("test/table/iris") |> DataFrame
        data = Array(iris[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])

        @testset "MCAR $r" for r in (0.15, 0.25, 0.35)
            # Original authors said they ran the test 1000 times
            results = map(1:1000) do i
                X = add_missings(data, r)
                knn_imputed = impute(copy(X), Impute.KNN(; k=3); dims=:cols)
                sub_imputed = impute(copy(X), Substitute(); dims=:cols)

                return nrmsd(data, knn_imputed), nrmsd(data, sub_imputed)
            end

            knn_nrmsd = first.(results)
            sub_nrmsd = last.(results)

            # Test that knn is better on average
            @test mean(knn_nrmsd) < mean(sub_nrmsd)
            # Test that the difference is "significant".
            @test pvalue(UnequalVarianceTTest(knn_nrmsd, sub_nrmsd)) < 0.05

            # test type stability
            @test typeof(impute(add_missings(data, r), Impute.KNN(; k=3); dims=:cols)) == Matrix{Union{Float64, Missing}}
            @test typeof(impute(add_missings(data, r), Substitute(); dims=:cols)) == Matrix{Union{Float64, Missing}}
        end
    end

    # Test a case where we expect kNN to perform well (e.g., many variables, )
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

        X = add_missings(data')
        num_tests = 100

        knn_nrmsd, mean_nrmsd = 0.0, 0.0

        for i = 1:num_tests
            knn_imputed = impute(copy(X), Impute.KNN(; k=4); dims=:cols)
            mean_imputed = impute(copy(X), Substitute(); dims=:cols)

            knn_nrmsd = ((i - 1) * knn_nrmsd + nrmsd(knn_imputed, data')) / i
            mean_nrmsd = ((i - 1) * mean_nrmsd + nrmsd(mean_imputed, data')) / i
        end

        # @show knn_nrmsd mean_nrmsd
        @test knn_nrmsd < mean_nrmsd
        # test type stability
        @test typeof(X) == typeof(impute(copy(X), Impute.KNN(; k=4); dims=:cols))
        @test typeof(X) == typeof(impute(copy(X), Substitute(); dims=:cols))
    end
end
