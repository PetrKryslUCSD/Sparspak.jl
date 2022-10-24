module GaloisTest
using Test
using LinearAlgebra
using SparseArrays
using Sparspak
using Sparspak.SpkProblem: insparse!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, solve!
using GaloisFields



function _test(T,n)
    spm = sprand(T, n, n, 1/n)
    for i=1:n
        spm[i,i]=one(T)
    end
    x=rand(T,n)
    b=spm*x
    p = Sparspak.SpkProblem.Problem(n, n, nnz(spm), zero(T))
    Sparspak.SpkProblem.insparse!(p, spm);
    Sparspak.SpkProblem.infullrhs!(p, b);
    s = SparseSolver(p)
    solve!(s)
    @test x==p.x
end

Base.typemax(::Type{T}) where T<:GaloisFields.AbstractGaloisField=one(T)

const F17=@GaloisField 17

_test(F17, 10)
_test(F17, 100)

const F127=@GaloisField 127
_test(F127, 10)

end

