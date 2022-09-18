using Test

@time @testset "Utilities" begin
    include("test_utilities.jl")
end

@time @testset "Problem" begin
    include("test_problem.jl")
end

@time @testset "Ordering" begin
    include("test_ordering.jl")
end

@time @testset "Graph" begin
    include("test_graph.jl")
end

@time @testset "ETree" begin
    include("test_etree.jl")
end

@time @testset "Grid" begin
    include("test_grid.jl")
end

@time @testset "Generic LAPACK/BLAS" begin
    include("test_blfragments.jl")
end

@time @testset "Sparse method" begin
    include("test_sparse_method.jl")
end

@time @testset "Generic floating type" begin
    include("test_generic.jl")
end

