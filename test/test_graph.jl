using Test

module morde001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkGraph

function _test()
    M, N = 21, 21
    p = SpkProblem.Problem(M, N)
    spm = sprand(M, N, 0.2)
    spm = spm + spm'
    SpkProblem.insparse(p, spm)
    graph = SpkGraph.Graph(p)
    @show graph
    return true
end

_test()
end # module

