module generic001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak
using Sparspak.SpkProblem: insparse!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, solve!
using Random
using MultiFloats, ForwardDiff

Random.rand(rng::AbstractRNG, ::Random.SamplerType{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = ForwardDiff.Dual{T,V,N}(rand(rng,T))

function makerandomproblem(T, n)
    spm = sprand(T, n, n, 1/n)
    spm = -spm - spm' + 20 * LinearAlgebra.I
    
    p = Sparspak.SpkProblem.Problem(n, n, nnz(spm), zero(T))
    Sparspak.SpkProblem.insparse!(p, spm);
    Sparspak.SpkProblem.infullrhs!(p, rand(T,n));
    return p
end

function _test(T)
    @info "testing $T"
    p = makerandomproblem(T,301)
    
    s = SparseSolver(p)
    solve!(s)
    A = Matrix(outsparse(p))
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6

    return true
end


_test(MultiFloats.Float64x2)
_test(ForwardDiff.Dual{Float64,Float64,1})
end # module
