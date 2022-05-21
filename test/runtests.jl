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
