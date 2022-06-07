using Test

module metre001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkETree

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    p = SpkProblem.Problem(M, N)
    SpkProblem.insparse!(p, spm)
    t = SpkETree.ETree(M)
    @test t.parent == [0, 0, 0, 0, 0, 0] 
    return true
end

_test()
end # module

