
struct ImputorTester{I<:Imputor}
    imp::Type{I}
    f::Function
    f!::Function
    kwargs::Dict
end

function ImputorTester(imp::Type{<:Imputor}; kwargs...)
    fname = lowercase(string(nameof(imp)))
    settings = Dict(kwargs...)
    settings[:context] = Context(; limit=1.0)

    return ImputorTester(
        imp,
        getfield(Impute, Symbol(fname)),
        getfield(Impute, Symbol(fname * "!")),
        settings,
    )
end

function test_all(tester::ImputorTester)
    test_equality(tester)
    test_vector(tester)
    test_matrix(tester)
    test_dataframe(tester)
    test_axisarray(tester)
    test_columntable(tester)
    # test_rowtable(tester)
end

function test_equality(tester::ImputorTester)
    @testset "Equality" begin
        @test tester.imp() == tester.imp()
    end
end

function test_vector(tester::ImputorTester)
    @testset "Vector" begin
        if tester.imp != DropVars
            a = allowmissing(1.0:1.0:20.0)
            a[[2, 3, 7]] .= missing

            result = impute(a, tester.imp(; tester.kwargs...))

            @testset "Base" begin
                # Test that we have fewer missing values
                @test count(ismissing, result) < count(ismissing, a)
                @test isa(result, Vector)
                @test eltype(result) <: eltype(a)

                # Test that functional form behaves the same way
                @test result == tester.f(a; tester.kwargs...)
            end

            @testset "In-place" begin
                # Test that the in-place function return the new results and logs whether it
                # successfully did it in-place
                a2 = deepcopy(a)
                a2_ = tester.f!(a2; tester.kwargs...)
                @test a2_ == result
                if a2 != result
                    @warn "$(tester.f!) did not mutate input data of type Vector"
                end
            end

            @testset "No missing" begin
                # Test having no missing data
                b = allowmissing(1.0:1.0:20.0)
                @test impute(b, tester.imp(; tester.kwargs...)) == b
            end

            @testset "All missing" begin
                # Test having only missing data
                c = fill(missing, 10)
                if tester.imp != Impute.DropObs
                    @test isequal(impute(c, tester.imp(; tester.kwargs...)), c)
                else
                    @test impute(c, tester.imp(; tester.kwargs...)) == empty(c)
                end
            end

            @testset "Too many missing values" begin
                # Test Context error condition
                c = fill(missing, 10)
                settings = deepcopy(tester.kwargs)
                settings[:context] = Context(; limit=0.1)
                @test_throws ImputeError impute(c, tester.imp(; settings...))
                @test_throws ImputeError tester.f(c; settings...)
            end
        end
    end
end

function test_matrix(tester::ImputorTester)
    @testset "Matrix" begin
        a = allowmissing(1.0:1.0:20.0)
        a[[2, 3, 7]] .= missing
        m = collect(reshape(a, 5, 4))

        result = impute(m, tester.imp(; tester.kwargs...))

        @testset "Base" begin
            # Test that we have fewer missing values
            @test count(ismissing, result) < count(ismissing, m)
            @test isa(result, Matrix)
            @test eltype(result) <: eltype(m)

            # Test that functional form behaves the same way
            @test result == tester.f(m; tester.kwargs...)
        end

        @testset "In-place" begin
            # Test that the in-place function return the new results and logs whether it
            # successfully did it in-place
            m2 = deepcopy(m)
            m2_ = tester.f!(m2; tester.kwargs...)
            @test m2_ == result
            if m2 != result
                @warn "$(tester.f!) did not mutate input data of type Matrix"
            end
        end

        @testset "No missing" begin
            # Test having no missing data
            b = collect(reshape(allowmissing(1.0:1.0:20.0), 5, 4))
            @test impute(b, tester.imp(; tester.kwargs...)) == b
        end

        @testset "All missing" begin
            # Test having only missing data
            c = fill(missing, 5, 2)
            if tester.imp == DropObs
                @test impute(c, tester.imp(; tester.kwargs...)) == Matrix{Missing}(missing, 0, 2)
            elseif tester.imp == DropVars
                @test impute(c, tester.imp(; tester.kwargs...)) == Matrix{Missing}(missing, 5, 0)
            else
                @test isequal(impute(c, tester.imp(; tester.kwargs...)), c)
            end
        end

        @testset "Too many missing values" begin
            # Test Context error condition
            c = fill(missing, 5, 2)
            settings = deepcopy(tester.kwargs)
            settings[:context] = Context(; limit=0.1)
            @test_throws ImputeError impute(c, tester.imp(; settings...))
            @test_throws ImputeError tester.f(c; settings...)
        end
    end
end

function test_dataframe(tester::ImputorTester)
    @testset "DataFrame" begin
        table = DataFrame(
            :sin => allowmissing(sin.(1.0:1.0:20.0)),
            :cos => allowmissing(sin.(1.0:1.0:20.0)),
        )

        table.sin[[2, 3, 7, 12, 19]] .= missing

        result = impute(table, tester.imp(; tester.kwargs...))

        @testset "Base" begin
            # Test that we have fewer missing values
            @test count(ismissing, Matrix(result)) < count(ismissing, Matrix(table))
            @test isa(result, DataFrame)

            # Test that functional form behaves the same way
            @test result == tester.f(table; tester.kwargs...)
        end

        @testset "In-place" begin
            # Test that the in-place function return the new results and logs whether it
            # successfully did it in-place
            table2 = deepcopy(table)
            table2_ = tester.f!(table2; tester.kwargs...)
            @test table2_ == result
            if table2 != result
                @warn "$(tester.f!) did not mutate input data of type DataFrame"
            end
        end

        @testset "No missing" begin
            # Test having no missing data
            b = DataFrame(
                :sin => allowmissing(sin.(1.0:1.0:20.0)),
                :cos => allowmissing(sin.(1.0:1.0:20.0)),
            )
            @test impute(b, tester.imp(; tester.kwargs...)) == b
        end

        @testset "All missing" begin
            # Test having only missing data
            c = DataFrame(
                :sin => fill(missing, 10),
                :cos => fill(missing, 10),
            )
            if tester.imp == DropObs
                @test impute(c, tester.imp(; tester.kwargs...)) == DataFrame()
            elseif tester.imp == DropVars
                # https://github.com/JuliaData/Tables.jl/issues/117
                @test_broken impute(c, tester.imp(; tester.kwargs...)) == DataFrame()
            else
                @test isequal(impute(c, tester.imp(; tester.kwargs...)), c)
            end
        end

        @testset "Too many missing values" begin
            # Test Context error condition
            c = DataFrame(
                :sin => fill(missing, 10),
                :cos => fill(missing, 10),
            )
            settings = deepcopy(tester.kwargs)
            settings[:context] = Context(; limit=0.1)
            if tester.imp == DropVars
                # https://github.com/JuliaData/Tables.jl/issues/117
                @test_throws MethodError impute(c, tester.imp(; settings...))
                @test_throws MethodError tester.f(c; settings...)
            else
                @test_throws ImputeError impute(c, tester.imp(; settings...))
                @test_throws ImputeError tester.f(c; settings...)
            end
        end
    end
end

function test_axisarray(tester::ImputorTester)
    @testset "AxisArray" begin
        a = allowmissing(1.0:1.0:20.0)
        a[[2, 3, 7]] .= missing
        m = collect(reshape(a, 5, 4))
        aa = AxisArray(
            deepcopy(m),
            Axis{:time}(DateTime(2017, 6, 5, 5):Hour(1):DateTime(2017, 6, 5, 9)),
            Axis{:id}(1:4)
        )
        result = impute(aa, tester.imp(; tester.kwargs...))

        @testset "Base" begin
            # Test that we have fewer missing values
            @test count(ismissing, result) < count(ismissing, aa)
            @test isa(result, AxisArray)
            @test eltype(result) <: eltype(aa)

            # Test that functional form behaves the same way
            @test result == tester.f(aa; tester.kwargs...)
        end

        @testset "In-place" begin
            # Test that the in-place function return the new results and logs whether it
            # successfully did it in-place
            aa2 = deepcopy(aa)
            aa2_ = tester.f!(aa2; tester.kwargs...)
            @test aa2_ == result
            if aa2 != result
                @warn "$(tester.f!) did not mutate input data of type AxisArray"
            end
        end
    end
end

function test_columntable(tester::ImputorTester)
    @testset "Column Table" begin
        table = DataFrame(
            :sin => allowmissing(sin.(1.0:1.0:20.0)),
            :cos => allowmissing(sin.(1.0:1.0:20.0)),
        )

        table.sin[[2, 3, 7, 12, 19]] .= missing
        coltab = Tables.columntable(table)

        result = impute(coltab, tester.imp(; tester.kwargs...))

        @testset "Base" begin
            # Test that we have fewer missing values
            @test count(ismissing, Tables.matrix(result)) < count(ismissing, Tables.matrix(coltab))
            @test isa(result, NamedTuple)

            # Test that functional form behaves the same way
            @test result == tester.f(coltab; tester.kwargs...)
        end

        @testset "In-place" begin
            # Test that the in-place function return the new results and logs whether it
            # successfully did it in-place
            coltab2 = deepcopy(coltab)
            coltab2_ = tester.f!(coltab2; tester.kwargs...)
            @test coltab2_ == result
            if coltab2 != result
                @warn "$(tester.f!) did not mutate input data of column table"
            end
        end
    end
end

# Row table tests will fail because vector input could be a row table or a vector of scalars
# function test_rowtable(tester::ImputorTester)
#     @testset "Row Table" begin
#         table = DataFrame(
#             :sin => allowmissing(sin.(1.0:1.0:20.0)),
#             :cos => allowmissing(sin.(1.0:1.0:20.0)),
#         )

#         table.sin[[2, 3, 7, 12, 19]] .= missing
#         rowtab = Tables.rowtable(table)

#         result = impute(rowtab, tester.imp(; tester.kwargs...))

#         @testset "Base" begin
#             # Test that we have fewer missing values
#             @test count(ismissing, Tables.matrix(result)) < count(ismissing, Tables.matrix(rowtab))
#             @test isa(result, Vector)

#             # Test that functional form behaves the same way
#             @test result == tester.f(rowtab; tester.kwargs...)
#         end

#         @testset "In-place" begin
#             # Test that the in-place function return the new results and logs whether it
#             # successfully did it in-place
#             rowtab2 = deepcopy(rowtab)
#             rowtab2_ = tester.f!(rowtab2; tester.kwargs...)
#             @test rowtab2_ == result
#             if rowtab2 != result
#                 @warn "$(tester.f!) did not mutate input data of row table"
#             end
#         end
#     end
# end
