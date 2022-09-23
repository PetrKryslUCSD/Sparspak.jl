using Test

module timing
using Test
using LinearAlgebra
using SparseArrays
using Sparspak
using Sparspak.SpkSparseSolver: SparseSolver, solve!
using Sparspak.SpkProblem: insparse!, outsparse
using BenchmarkTools, Random
using ForwardDiff

Random.rand(rng::AbstractRNG, ::Random.SamplerType{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = ForwardDiff.Dual{T,V,N}(rand(rng,T))


function solve1(spm, b)
    T=eltype(spm)
    n=size(spm,1)
    begin
        p = Sparspak.SpkProblem.Problem(n, n, nnz(spm), zero(T))
        Sparspak.SpkProblem.insparse!(p, spm);
        Sparspak.SpkProblem.infullrhs!(p, b);
        s = SparseSolver(p)
    end
    solve!(s)
    p.x
end






function _test(;T=Float64, n=20)
    Random.seed!(9876)
    spm = sprand(T, n, n, 1/n)
    spm = -spm - spm' + 40 * LinearAlgebra.I
    b=rand(T,n)

    @btime solve1($spm,$b)
    nothing
end


end



