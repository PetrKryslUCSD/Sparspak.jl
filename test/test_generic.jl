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
    spm = -spm - spm' + 40 * LinearAlgebra.I
    
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


module generic002
using  Test
using Tensors
using LinearAlgebra
using Sparspak
using SparseArrays
using ForwardDiff

function tridiagonal(p,n)
    T=typeof(p)
    b=T[p^i for i=1:n]
    a=fill(T(-0.1),n-1)
    c=fill(T(-0.1),n-1)
    Tridiagonal(a,b,c)
end

function f(p)
    n=20
    M=Matrix(tridiagonal(p,n))
    f=ones(n)
    sum(M\f)
end

df(x)=Tensors.gradient(f,x)

function g(p)
    n=20
    M=sparse(tridiagonal(p,n))
    f=ones(n)
    pr = Sparspak.SpkProblem.Problem(n,n,nnz(M),zero(p))
    Sparspak.SpkProblem.insparse!(pr, M)
    Sparspak.SpkProblem.infullrhs!(pr,f)
    s = Sparspak.SparseSolver.SparseSolver(pr)
    Sparspak.SparseSolver.solve!(s)    
    sum(pr.x)
end

dg(x)=Tensors.gradient(g,x)


function _test()
    @info df(4.0), dg(4.0)
    @test df(4.0)≈dg(4.0)
    true
end

_test()
end # module
