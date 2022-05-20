using Test

@time @testset "Utilities" begin
    include("test_utilities.jl")
end

@time @testset "Problem" begin
    include("test_problem.jl")
end