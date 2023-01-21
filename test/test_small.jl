
using Test
using Sparspak
using Random, SparseArrays, LinearAlgebra
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkSparseSolver

#
# This is so far the simplest test problem occuring
#
function simpletest(;n=4)
    A = sparse(Diagonal(ones(n)))
    A[2,1]=-0.1
    pr = SpkProblem.Problem(n,n, 10)
    @test SpkProblem.insparse!(pr, A)
    @show pr
    s = SpkSparseSolver.SparseSolver(pr)
    @test SpkSparseSolver.findorder!(s)
end

simpletest()

#=

julia> include("C:\\Users\\pkonl\\Documents\\00WIP\\Sparspak.jl\\test\\test_small.jl")pr = Sparspak.SpkProblem.Problem{Int64, Float64}("", 4, 10, 4, 5, [1, 3, 4, 5], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 2, 3, 4, 0, 0, 0, 0, 0], 4, 4, 15, 4, 5, 4, [1.00000e+00, -1.00000e-01, 1.00000e+00, 1.00000e+00, 1.00000e+00, Inf, Inf, Inf, Inf, Inf], Float64[], Float64[], [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00], [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00])  

g = Sparspak.SpkGraph.Graph{Int64}(4, 1, 4, 4, [1, 2, 2, 2, 2], [2])                  
(k, first, g.adj) = (2, [1, 2, 2, 2], [2])  

=#
