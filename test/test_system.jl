using Test

module msyst001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij, inbi, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, solve

function maketridiagproblem(n)
    p = SpkProblem.Problem(n, n)
    for i in 1:(n-1)
        inaij(p, i + 1, i, -1.0)
        inaij(p, i, i, 4.0)
        inaij(p, i, i + 1, -1.0)
        inbi(p, i, 1.0)
    end
    inaij(p, n, n, 4.0)
    inbi(p, n, 1.0)
    return p
end

function _test()
    p = maketridiagproblem(11)
    # @show outsparse(p)
    s = SparseSolver(p)
    solve(s, p)
    return true
end

_test()
end # module

