module generic001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak
using Sparspak.SpkProblem: insparse!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, solve!
using MultiFloats

function makerandomproblem(T, n)
    spm = sprand(T, n, n, 1/n)
#    spm=sparse(ones(n,n))
    spm = -spm - spm' + 20 * LinearAlgebra.I
    
    p = Sparspak.SpkProblem.Problem(n, n, nnz(spm), zero(T))
    Sparspak.SpkProblem.insparse!(p, spm);
    Sparspak.SpkProblem.infullrhs!(p, rand(T,n));
    return p
end

function _test(T)
    p = makerandomproblem(T,301)
    
    s = SparseSolver(p)
    solve!(s)
    A = Matrix(outsparse(p))
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6

    return true
end

_test(Float64)
#_test(MultiFloats.Float64x2)
end # module
