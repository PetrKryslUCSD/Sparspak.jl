module GaloisTest
using Test
using LinearAlgebra
using SparseArrays
using Sparspak
using Sparspak.SpkProblem: insparse!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, solve!
using GaloisFields

@static if VERSION < v"1.9"
    Sparspak.GenericBlasLapackFragments.lupivottype(::Type{T}) where T<:GaloisFields.AbstractGaloisField= Sparspak.GenericBlasLapackFragments.RowNonZero()
else
    LinearAlgebra.lupivottype(::Type{T}) where T<:GaloisFields.AbstractGaloisField= LinearAlgebra.RowNonZero()
end

Sparspak.SpkUtilities._BIGGY(::Type{T}) where T<:GaloisFields.AbstractGaloisField=zero(T)
Base.abs(x::T) where T<:GaloisFields.AbstractGaloisField=x
Base.isless(x::T,y::T) where T<:GaloisFields.AbstractGaloisField=x.n<y.n



function _test(T,n)
    # need some scalable (random ?) invertible test matrices here
    spm0 = sprand(Int8, n, n, 1/n)
    spm0 = -spm0 - spm0' + 40 * LinearAlgebra.I
    spm=SparseMatrixCSC(n,n,spm0.colptr,spm0.rowval,T.(spm0.nzval))
    @show typeof(spm)
    x=rand(T,n)
    b=spm*x
    p = Sparspak.SpkProblem.Problem(n, n, nnz(spm), zero(T))
    Sparspak.SpkProblem.insparse!(p, spm);
    Sparspak.SpkProblem.infullrhs!(p, b);
    s = SparseSolver(p)
    solve!(s)
    @test x == p.x
end


const F1013=@GaloisField 1013

 _test(F1013, 100)

const F127=@GaloisField 127
 _test(F127, 10)


end

